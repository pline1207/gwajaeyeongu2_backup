import math
import numpy as np
from pathlib import Path
import random
from functools import partial
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.optim import Adam

from einops import rearrange, reduce
# from scipy.optimize import linear_sum_assignment # 사용 안 함

from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.utils.tensorboard import SummaryWriter

from .layers import SHConv, SHT, ISHT
from ..lib.sphere import vertex_area, spharm_real
from ..lib.io import read_mesh
from ..lib.utils import mean_curvature
from ..lib.loss import lap_loss, normal_consistency

from .ViViT.vivit import ViViT 

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 


# --- Helper Functions ---
def identity(t, *args, **kwargs):
    return t

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# --- Beta Schedules (기존과 동일) ---
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def spherical_collate_longitudinal(batch):
    # (기존과 동일)
    xs, names, time_obs_list, subject_id_list = [], [], [], []
    refs = [] 

    for item in batch:
        d = item[0]
        n = item[1]
        t_obs = item[2]
        s_id = item[3]
        
        if not isinstance(d, torch.Tensor):
            d = torch.from_numpy(d)
        xs.append(d.float())
        names.append(n)
        time_obs_list.append(t_obs)
        subject_id_list.append(s_id)

        if len(item) > 4:
            r = item[4]
            if not isinstance(r, torch.Tensor):
                r = torch.from_numpy(r)
            refs.append(r.float())

    x = torch.stack(xs, dim=0)   
    T_obs = torch.tensor(time_obs_list, dtype=torch.float32) 
    S_id = torch.tensor(subject_id_list, dtype=torch.long)  
    
    if len(refs) > 0:
        ref_batch = torch.stack(refs, dim=0)
        return (x, T_obs, S_id, ref_batch, names)
    else:
        return (x, T_obs, S_id, names)


class SinusoidalPosEmb(Module):
    # (기존과 동일)
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SHConvBlock(nn.Module):
    # (기존과 동일)
    def __init__(self, Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear=None, fullband=True, bn=True): 
        super().__init__()
        self.shconv = nn.Sequential(SHT(L, Y_inv, area), SHConv(in_ch, out_ch, L, interval), ISHT(Y))
        self.impulse = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, bias=not bn) if fullband else lambda _: 0
        self.bn = nn.BatchNorm1d(out_ch, momentum=0.1, affine=True, track_running_stats=False) if bn else nn.Identity()
        self.nonlinear = nonlinear if nonlinear is not None else nn.Identity()

    def forward(self, x, scale_shift=None):
        x = self.shconv(x) + self.impulse(x)
        x = self.bn(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.nonlinear(x)
        return x


class SHConvResnetBlock(nn.Module):
    # (기존과 동일)
    def __init__(self, Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear=None, fullband=True, bn=True, time_emb_dim=None, **kwargs):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2) 
        ) if time_emb_dim is not None else None

        self.block1 = SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, fullband, bn)
        self.block2 = SHConvBlock(Y, Y_inv, area, out_ch, out_ch, L, interval, nonlinear, fullband, bn)
        self.res_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1) if (in_ch != out_ch) else nn.Identity()
    
    def forward(self, x, time_emb=None):
        scale_shift = None
        if (self.mlp is not None) and (time_emb is not None):
            ss = self.mlp(time_emb)
            scale, shift = ss.chunk(2, dim=1)
            scale_shift = (scale.unsqueeze(-1), shift.unsqueeze(-1))
            
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h, scale_shift=None)
        return h + self.res_conv(x)


class SPHARM_Net(nn.Module):
    # (기존과 동일 - 건드릴 필요 없음)
    def __init__(self, sphere, device, in_ch=1, n_class=1, C=128, L=80, D=3, interval=5, 
                 sinusoidal_pos_emb_theta=10000, self_condition=False, verbose=False, add_xyz=True, 
                 max_time_obs=None, num_subjects=None,
                 use_ref_condition=False, ref_in_ch=1, ref_num_frames=None):
        super().__init__()
        v, f = read_mesh(sphere)
        v = v.astype(float)
        area = vertex_area(v, f)
        Y = spharm_real(v, L, threads=1)

        self.channels = in_ch
        self.out_dim  = n_class
        self.add_xyz = add_xyz

        area = torch.from_numpy(area).to(device=device, dtype=torch.float32)
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
        Y_inv = Y.T

        if self.add_xyz:
            pos = torch.from_numpy(v).to(device=device, dtype=torch.float32)
            pos = pos.t().unsqueeze(0)
            self.register_buffer("pos_xyz", pos)

        self.self_condition = self_condition
        input_ch = in_ch * (2 if self_condition else 1)
        input_ch = input_ch + (3 if self.add_xyz else 0)

        time_dim = C * 4
        self.time_pos_emb = SinusoidalPosEmb(C, theta=sinusoidal_pos_emb_theta)
        self.time_mlp = nn.Sequential(self.time_pos_emb, nn.Linear(C, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

        self.max_time_obs = max_time_obs
        if max_time_obs is not None:
            self.time_obs_pos_emb = SinusoidalPosEmb(C, theta=sinusoidal_pos_emb_theta)
            self.time_obs_mlp = nn.Sequential(self.time_obs_pos_emb, nn.Linear(C, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

        self.num_subjects = num_subjects
        if num_subjects is not None:
            self.subject_emb = nn.Embedding(num_subjects, time_dim) 
            nn.init.uniform_(self.subject_emb.weight, -0.01, 0.01)

        self.use_ref_condition = use_ref_condition
        self.ref_num_frames = ref_num_frames if ref_num_frames else 1
        
        if self.use_ref_condition:
            self.vivit = ViViT(
                image_size = L,          
                patch_size = 16,         
                num_classes = time_dim,  
                dim = 512,
                depth = 6,
                heads = 8,
                in_channels = ref_in_ch,
                num_frames = self.ref_num_frames 
            )
            
            self.vivit_image_size = L

            self.vivit_proj = nn.Sequential(
                nn.LayerNorm(time_dim),
                nn.Linear(time_dim, time_dim),
                nn.GELU()
            )

        combined_dim_in = time_dim 
        if max_time_obs is not None:
            combined_dim_in += time_dim 
        if num_subjects is not None:
            combined_dim_in += time_dim 
        if self.use_ref_condition:
            combined_dim_in += time_dim 

        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim_in, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim)
        ) if combined_dim_in > time_dim else nn.Identity()

 
        L_in = L
        ch_inc = 2
        out_ch = C

        self.down = nn.ModuleList()
        self.up   = nn.ModuleList()

        current_in = input_ch
        for i in range(D):
            L = L_in // (2 ** i)
            self.down.append(
                SHConvResnetBlock(Y, Y_inv, area, in_ch=current_in, out_ch=out_ch, L=L, interval=interval, nonlinear=F.relu, fullband=(i == 0), bn=True, time_emb_dim=time_dim)
            )
            current_in = out_ch
            out_ch *= ch_inc

        L //= 2
        out_ch //= ch_inc
        current_in = out_ch
        self.bottom = SHConvResnetBlock(Y, Y_inv, area, in_ch=current_in, out_ch=out_ch, L=L, interval=interval, nonlinear=F.relu, fullband=False, bn=True, time_emb_dim=time_dim)

        for i in range(D - 1):
            L = L_in // (2 ** (D - 1 - i))
            in_ch_up = out_ch * 2
            out_ch //= ch_inc
            self.up.append(
                SHConvResnetBlock(Y, Y_inv, area, in_ch=in_ch_up, out_ch=out_ch, L=L, interval=interval, nonlinear=F.relu, fullband=False, bn=True, time_emb_dim=time_dim)
            )

        in_ch_up = out_ch * 2
        L *= 2
        self.up.append(
            SHConvResnetBlock(Y, Y_inv, area, in_ch=in_ch_up, out_ch=out_ch, L=L, interval=interval, nonlinear=F.relu, fullband=True, bn=True, time_emb_dim=time_dim)
        )

        self.final = SHConvBlock(Y, Y_inv, area, in_ch=out_ch, out_ch=n_class, L=L_in, interval=interval, nonlinear=None, fullband=True, bn=True)

    def forward(self, x, t=None, x_self_cond=None, time_obs=None, subject_id=None, ref_img=None):
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat([x_self_cond, x], dim=1) 

        if self.add_xyz:
            B = x.shape[0]
            x = torch.cat([x, self.pos_xyz.expand(B, -1, -1)], dim=1)

        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        diff_time_emb = self.time_mlp(t) 

        cond_embs = [diff_time_emb]

        if self.max_time_obs is not None and time_obs is not None:
            norm_time_obs = time_obs / self.max_time_obs 
            time_obs_emb = self.time_obs_mlp(norm_time_obs)
            cond_embs.append(time_obs_emb)

        if self.num_subjects is not None and subject_id is not None:
            subject_emb = self.subject_emb(subject_id) 
            cond_embs.append(subject_emb)

        if self.use_ref_condition:
            if ref_img is not None:
                x_ref = ref_img
                
                target_size = self.vivit.image_size 
                target_area = target_size * target_size
                
                if x_ref.ndim == 3:
                    x_ref = x_ref.unsqueeze(1)

                b, t_ref, c, n = x_ref.shape
                x_ref_flat = x_ref.view(b * t_ref, c, n)

                if n != target_area:
                    x_ref_flat = F.interpolate(x_ref_flat, size=target_area, mode='linear', align_corners=False)
                
                x_ref_2d = x_ref_flat.view(b * t_ref, c, target_size, target_size)
                x_ref_final = x_ref_2d.view(b, t_ref, c, target_size, target_size)
                
                ref_emb = self.vivit(x_ref_final)
                ref_emb = self.vivit_proj(ref_emb)
                cond_embs.append(ref_emb)
            else:
                cond_embs.append(torch.zeros_like(diff_time_emb))

        if len(cond_embs) > 1:
            combined_emb = torch.cat(cond_embs, dim=1) 
            time_emb = self.combined_mlp(combined_emb) 
        else:
            time_emb = diff_time_emb 

        feats = []
        y = x
        for blk in self.down:
            y = blk(y, time_emb=time_emb)
            feats.append(y)

        y = self.bottom(y, time_emb=time_emb)

        for i, blk in enumerate(self.up):
            skip = feats[-1 - i]
            y = torch.cat([y, skip], dim=1)
            y = blk(y, time_emb=time_emb)

        y = self.final(y) 
        return y


class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        signal_size: int = None,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  
        min_snr_loss_weight = False, 
        min_snr_gamma = 5,
        immiscible = False,
        # geometry supervision
        faces = None,                    
        edge_face_indices = None,        
        lambda_cyc: float = 0.0,        
        curvature_loss: str = 'l1',     
        lambda_lap: float = 0.0,        
        lambda_normal: float = 0.0      
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.signal_size = signal_size
        self.objective = objective

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.immiscible = immiscible
        self.offset_noise_strength = offset_noise_strength

        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        self.lambda_cyc    = float(lambda_cyc)
        self.curvature_loss = curvature_loss
        self.lambda_lap    = float(lambda_lap)
        self.lambda_normal= float(lambda_normal)
        self.faces = faces
        self.edge_face_indices = edge_face_indices

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 🌟 [수정 1] Classifier-Free Guidance (CFG) Logic Implementation
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, 
                          time_obs = None, subject_id = None, ref_img = None, 
                          cond_scale = 1., **kwargs): # cond_scale 인자 추가
        
        # CFG 적용: cond_scale이 1보다 크고 reference가 있을 때만
        if cond_scale != 1. and ref_img is not None:
            # 1. Conditioned Prediction (Reference O)
            model_output_cond = self.model(x, t, x_self_cond, time_obs=time_obs, subject_id=subject_id, ref_img=ref_img, **kwargs)
            
            # 2. Unconditioned Prediction (Reference X, Zero Tensor)
            # null_ref는 0으로 채워진 텐서 (Unconditional 학습 시와 동일한 조건)
            null_ref = torch.zeros_like(ref_img) 
            model_output_uncond = self.model(x, t, x_self_cond, time_obs=time_obs, subject_id=subject_id, ref_img=null_ref, **kwargs)
            
            # 3. CFG Formula: Uncond + Scale * (Cond - Uncond)
            model_output = model_output_uncond + cond_scale * (model_output_cond - model_output_uncond)
        else:
            # 일반적인 경우 (또는 Ref가 없을 때)
            model_output = self.model(x, t, x_self_cond, time_obs=time_obs, subject_id=subject_id, ref_img=ref_img, **kwargs)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = maybe_clip(model_output)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True, 
                        time_obs = None, subject_id = None, ref_img = None, 
                        cond_scale = 1., **kwargs):
        # cond_scale 전달
        preds = self.model_predictions(x, t, x_self_cond, time_obs=time_obs, subject_id=subject_id, ref_img=ref_img, cond_scale=cond_scale, **kwargs)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start = x_start, x_t = x, t = t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, 
                 time_obs = None, subject_id = None, ref_img = None, 
                 cond_scale = 1., **kwargs):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True, 
            time_obs=time_obs, subject_id=subject_id, ref_img=ref_img, 
            cond_scale=cond_scale, **kwargs
        )
        noise = torch.randn_like(x) if t > 0 else 0
        pred = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False, 
                      time_obs = None, subject_id = None, ref_img = None, 
                      cond_scale = 1., **kwargs):
        batch, device = shape[0], self.device
        img = torch.randn(shape, device = device)
        imgs = [img]
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            # cond_scale 전달
            img, x_start = self.p_sample(img, t, self_cond, 
                                         time_obs=time_obs, subject_id=subject_id, ref_img=ref_img, 
                                         cond_scale=cond_scale, **kwargs)
            imgs.append(img)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        return self.unnormalize(ret)

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps = False, 
                    time_obs = None, subject_id = None, ref_img = None, 
                    cond_scale = 1., **kwargs):
        batch, device = shape[0], self.device
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device = device)
        imgs = [img]
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            # cond_scale 전달
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, 
                time_obs=time_obs, subject_id=subject_id, ref_img=ref_img, 
                cond_scale=cond_scale, **kwargs
            )

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        return self.unnormalize(ret)

    @torch.no_grad()
    def sample(self, batch_size = 16, sample_size: int = None, return_all_timesteps = False, 
               time_obs = None, subject_id = None, ref_img = None, 
               cond_scale = 1.): # cond_scale 기본값 1.0 (CFG 미적용)
        channels = self.channels
        N = sample_size if sample_size is not None else self.signal_size
        if N is None:
            raise ValueError("Please provide `sample_size` or set `signal_size` in GaussianDiffusion(...)")

        shape = (batch_size, channels, N)
        # cond_scale을 kwargs에 포함
        kwargs = dict(time_obs=time_obs, subject_id=subject_id, ref_img=ref_img, cond_scale=cond_scale)
        fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return fn(shape, return_all_timesteps = return_all_timesteps, **kwargs)

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if self.immiscible:
            pass
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None, 
                 time_obs = None, subject_id = None, ref_img = None): 
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)
        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * offset_noise.unsqueeze(-1)

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, time_obs=time_obs, subject_id=subject_id, ref_img=ref_img).pred_x_start
                x_self_cond.detach_()

        model_out = self.model(x, t, x_self_cond, time_obs=time_obs, subject_id=subject_id, ref_img=ref_img)

        if self.objective == 'pred_noise':
            target = noise
            x0_hat = self.predict_start_from_noise(x, t, model_out)
        elif self.objective == 'pred_x0':
            target = x_start
            x0_hat = model_out
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
            x0_hat = self.predict_start_from_v(x, t, model_out)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        loss = loss.mean()

        if (self.faces is not None) and (x0_hat.shape[1] >= 4):
            curv_hat = x0_hat[:, 0:1, :]
            xyz_hat  = x0_hat[:, 1:4, :]
            verts = xyz_hat.transpose(1, 2).contiguous()

            if self.lambda_cyc > 0.0:
                _, H, _ = mean_curvature(verts, self.faces)
                curv_pred = curv_hat.squeeze(1)
                if self.curvature_loss.lower() == 'l2':
                    cyc = F.mse_loss(H, curv_pred, reduction='mean')
                else:
                    cyc = F.l1_loss(H, curv_pred, reduction='mean')
                loss = loss + self.lambda_cyc * cyc

            if self.lambda_lap > 0.0:
                lap = lap_loss(verts, self.faces).mean()
                loss = loss + self.lambda_lap * lap

            if (self.lambda_normal > 0.0) and (self.edge_face_indices is not None):
                nerr = normal_consistency(verts, self.faces, self.edge_face_indices).mean()
                loss = loss + self.lambda_normal * nerr

        return loss

    def forward(self, data_batch, *args, **kwargs):
        try:
            img = data_batch[0]
            time_obs = data_batch[1]
            subject_id = data_batch[2]
            
            if len(data_batch) > 3:
                candidate = data_batch[3]
                if isinstance(candidate, torch.Tensor):
                    ref_img = candidate
                else:
                    ref_img = None
            else:
                ref_img = None
            
            if time_obs is not None: time_obs = time_obs.to(img.device)
            if subject_id is not None: subject_id = subject_id.to(img.device)
            if ref_img is not None: ref_img = ref_img.to(img.device)

        except (TypeError, IndexError):
            img = data_batch[0] if isinstance(data_batch, (tuple, list)) else data_batch
            time_obs = None
            subject_id = None
            ref_img = None

        b, device = img.shape[0], img.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = self.normalize(img)

        return self.p_losses(img, t, *args, 
                             time_obs=time_obs, 
                             subject_id=subject_id,
                             ref_img=ref_img,
                             **kwargs)


class SphericalTrainer:
    def __init__(
        self, diffusion_model, dataset, *, train_batch_size = 16, gradient_accumulate_every = 1, train_lr = 1e-4, train_num_steps = 100_000, ema_update_every = 10, ema_decay = 0.995,
        adam_betas = (0.9, 0.99), save_and_sample_every = 1000, num_samples = 16, results_folder = './results_sph', split_batches = True,
        max_grad_norm = 1.0, save_best_and_latest_only = False, sphere_path: str = None, log_dir = './logs', valid_dataset = None
    ):
        super().__init__()
        self.model = diffusion_model.to(diffusion_model.device)
        self.channels = self.model.channels
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.ds = dataset
        self.dl = DataLoader(
            self.ds,
            batch_size = train_batch_size,
            shuffle = True,
            collate_fn = spherical_collate_longitudinal
        )
        self.data_iter = self._cycle(self.dl)
        
        self.valid_dl = None
        if valid_dataset is not None:
            self.valid_dl = DataLoader(
                valid_dataset,
                batch_size = train_batch_size,
                shuffle = False,
                collate_fn = spherical_collate_longitudinal
            )

        self.opt = Adam(self.model.parameters(), lr = train_lr, betas = adam_betas)
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to(diffusion_model.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.loss_history = []
        self.step = 0
        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.model.device

    def _cycle(self, dl):
        while True:
            for b in dl:
                yield b

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict() if hasattr(self, 'ema') else None,
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        device = self.model.device
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if hasattr(self, 'ema') and data.get('ema') is not None:
            self.ema.load_state_dict(data['ema'])

    def train(self):
        device = self.model.device
        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                total_loss = 0.0
                current_batch_ids = [] 

                for _ in range(self.gradient_accumulate_every):
                    data_batch = next(self.data_iter)
                    current_names = data_batch[-1] if isinstance(data_batch[-1], list) else []
                    current_batch_ids.extend(current_names)
                    
                    data_batch_on_device = []
                    for item in data_batch:
                        if isinstance(item, torch.Tensor):
                            data_batch_on_device.append(item.to(device))
                        else:
                            data_batch_on_device.append(item)
                    
                    data_batch_on_device = tuple(data_batch_on_device)

                    with autocast(enabled=False):
                        loss = self.model(data_batch_on_device)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    loss.backward()
                    
                processed_ids = ", ".join(current_batch_ids[:5]) + ("..." if len(current_batch_ids)>5 else "")
                pbar.set_description(f'Loss: {total_loss:.4f} | Data: [{processed_ids}]')

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                
                self.loss_history.append(total_loss)
                if hasattr(self, 'writer'):
                    self.writer.add_scalar('Loss/train', total_loss, self.step)

                self.step += 1
                if hasattr(self, 'ema'):
                    self.ema.update()

                if self.step != 0 and (self.step % self.save_and_sample_every == 0):
                    self._sample_and_save()
                    milestone = self.step // self.save_and_sample_every
                    if self.save_best_and_latest_only:
                        self.save("latest")
                    else:
                        self.save(milestone)
                        
                    if len(self.loss_history) > 0:
                        try:
                            plt.figure(figsize=(12, 6))
                            plt.plot(self.loss_history)
                            plt.title(f'Training Loss (up to Step {self.step})')
                            plt.grid(True)
                            plt.savefig(self.results_folder / 'loss_graph_realtime.png')
                            plt.close() 
                        except Exception: pass
                        
                pbar.update(1)

        print('training complete')

    @torch.no_grad()
    def _sample_and_save(self):
        if not hasattr(self, 'ema'): return
        self.ema.ema_model.eval()

        milestone = self.step // self.save_and_sample_every
        
        loader = self.valid_dl if self.valid_dl is not None else self.dl
        try:
            batch = next(iter(loader))
        except StopIteration:
            batch = next(iter(loader))

        B = min(self.num_samples, batch[0].shape[0])
        
        real_target = batch[0][:B].to(self.device)
        time_obs = batch[1][:B].to(self.device)
        subject_id = batch[2][:B].to(self.device) 
        
        ref_img = None
        if len(batch) > 3: 
             ref_img = batch[3][:B].to(self.device)

        names_list = batch[4][:B] if len(batch) > 4 else []

        # 🌟 [수정] CFG 가중치 설정 (가중치 3.0을 주어 Reference를 강하게 반영)
        COND_SCALE = 3.0 
        
        print(f"\n--- Sampling Start (Milestone {milestone}) using REAL Data (Scale: {COND_SCALE}) ---")
        
        samples = self.ema.ema_model.sample(
            batch_size=B, 
            sample_size=real_target.shape[-1], 
            return_all_timesteps=False,
            time_obs=time_obs, 
            subject_id=subject_id,
            ref_img=ref_img,
            cond_scale=COND_SCALE # 가중치 전달
        )

        save_path_npy = self.results_folder / f'samples-{milestone}.npy'
        saved_data = np.stack([samples.detach().cpu().numpy(), real_target.cpu().numpy()], axis=0)
        np.save(save_path_npy, saved_data)
        
        print(f"Saved samples (Generated vs Real) to {save_path_npy}")

        if subject_id is not None and hasattr(self.ds, 'subject_to_id'):
            save_path_txt = self.results_folder / f'samples-{milestone}.txt'
            
            s_ids_list = subject_id.cpu().tolist()
            t_obs_list = time_obs.cpu().tolist()
            
            with open(save_path_txt, 'w') as f:
                f.write(f"Index\tSubject_Name\tAge(Time)\tRef_Used\tCFG_Scale\n")
                for i in range(B):
                    subj_name = names_list[i] if len(names_list) > i else str(s_ids_list[i])
                    t_val = t_obs_list[i]
                    ref_status = "Yes" if ref_img is not None else "No"
                    
                    f.write(f"{i}\t{subj_name}\t{t_val:.4f}\t{ref_status}\t{COND_SCALE}\n")
            
            print(f"Saved metadata to {save_path_txt}")
