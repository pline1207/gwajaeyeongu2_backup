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

from scipy.optimize import linear_sum_assignment

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

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def default(val, d):
    return val if val is not None else (d() if callable(d) else d)

def identity(x):
    return x

def extract(a, t, x_shape):
    """
    a: (T,) buffer
    t: (B,) long
    x_shape: e.g. (B, C, N)
    returns shape (B, 1, 1) to broadcast over (C, N)
    """
    b = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def to_device(batch, device):
    data_np, _name = batch
    if isinstance(data_np, torch.Tensor):
        x = data_np
    else:
        x = torch.from_numpy(data_np)
    x = x.to(device=device, dtype=torch.float32)
    return x, _name

def spherical_collate_longitudinal(batch):
    # batch: list of (data(C,N), name, time_obs(float), subject_id(int))
    xs, names, time_obs_list, subject_id_list = [], [], [], []
    refs = [] # Reference image list

    for item in batch:
        # item 구조가 (d, n, t_obs, s_id) 라고 가정 (기존)
        # 만약 (d, n, t_obs, s_id, ref) 라면 처리가 필요함
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

        # ref_img가 있다면 처리
        if len(item) > 4:
            r = item[4]
            if not isinstance(r, torch.Tensor):
                r = torch.from_numpy(r)
            refs.append(r.float())

    x = torch.stack(xs, dim=0)   # (B, C, N)
    T_obs = torch.tensor(time_obs_list, dtype=torch.float32) # (B,)
    S_id = torch.tensor(subject_id_list, dtype=torch.long)  # (B,)
    
    if len(refs) > 0:
        ref_batch = torch.stack(refs, dim=0)
        return (x, T_obs, S_id, ref_batch, names)
    else:
        return (x, T_obs, S_id, names)


class SinusoidalPosEmb(Module):
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
    def __init__(self, Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear=None, fullband=True, bn=True, time_emb_dim=None, max_time_obs = None, num_subjects = None, **kwargs):
        super().__init__()

        self.mlp=nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2) # (scale, shift)
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
    def __init__(self, sphere, device, in_ch=1, n_class=1, C=128, L=80, D=3, interval=5, 
                 sinusoidal_pos_emb_theta=10000, self_condition=False, verbose=False, add_xyz=True, 
                 max_time_obs=None, num_subjects=None,
                 # 🌟 [추가] ViViT 관련 인자
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

        # self-conditioning
        self.self_condition = self_condition
        input_ch = in_ch * (2 if self_condition else 1)
        input_ch = input_ch + (3 if self.add_xyz else 0)

        # time embedding
        time_dim = C * 4
        self.time_pos_emb = SinusoidalPosEmb(C, theta=sinusoidal_pos_emb_theta)
        self.time_mlp = nn.Sequential(self.time_pos_emb, nn.Linear(C, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

        # --- 2. Observation Time Embedding ---
        self.max_time_obs = max_time_obs
        if max_time_obs is not None:
            self.time_obs_pos_emb = SinusoidalPosEmb(C, theta=sinusoidal_pos_emb_theta)
            self.time_obs_mlp = nn.Sequential(self.time_obs_pos_emb, nn.Linear(C, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

        # --- 3. Subject ID Embedding ---
        self.num_subjects = num_subjects
        if num_subjects is not None:
            self.subject_emb = nn.Embedding(num_subjects, time_dim) 
            nn.init.uniform_(self.subject_emb.weight, -0.01, 0.01)

        # 🌟 [추가] 4. ViViT Reference Embedding ---
        self.use_ref_condition = use_ref_condition
        if self.use_ref_condition:
            # ViViT 초기화 (파라미터는 실제 메쉬 사이즈 및 vivit.py 구현에 맞게 조정 필요)
            # L: vertex count or image size analog
            self.vivit = ViViT(
                image_size = L,          # SPHARM L 또는 vertex 수
                patch_size = 16,         # 적절히 조정
                num_classes = time_dim,  # ViViT 출력을 time_dim에 맞춤
                dim = 512,
                depth = 6,
                heads = 8,
                mlp_dim = 1024,
                channels = ref_in_ch,
                frames = ref_num_frames if ref_num_frames else 1
            )
            # Projection head (optional)
            self.vivit_proj = nn.Sequential(
                nn.LayerNorm(time_dim),
                nn.Linear(time_dim, time_dim),
                nn.GELU()
            )

        # --- 5. Combined Embedding MLP ---
        # 3개의 임베딩(t, T_obs, S_id) + ViViT Ref Emb 합친 후 변환
        combined_dim_in = time_dim # t
        if max_time_obs is not None:
            combined_dim_in += time_dim # + T_obs
        if num_subjects is not None:
            combined_dim_in += time_dim # + S_id
        if self.use_ref_condition:
            combined_dim_in += time_dim # + Ref Emb

        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim_in, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim) # 최종 출력은 time_dim
        ) if combined_dim_in > time_dim else nn.Identity()

 
        L_in = L
        ch_inc = 2
        out_ch = C

        self.down = nn.ModuleList()
        self.up   = nn.ModuleList()

        # Encoding
        current_in = input_ch
        for i in range(D):
            L = L_in // (2 ** i)
            if verbose:
                print(f"Down {i+1}\t| C:{current_in} -> {out_ch}\t| L:{L}")
            self.down.append(
                SHConvResnetBlock(Y, Y_inv, area, in_ch=current_in, out_ch=out_ch, L=L, interval=interval, nonlinear=F.relu, fullband=(i == 0), bn=True, time_emb_dim=time_dim)
            )
            current_in = out_ch
            out_ch *= ch_inc

        # Bottom
        L //= 2
        out_ch //= ch_inc
        current_in = out_ch
        if verbose:
            print(f"Bottom\t| C:{current_in} -> {out_ch}\t| L:{L}")
        self.bottom = SHConvResnetBlock(Y, Y_inv, area, in_ch=current_in, out_ch=out_ch, L=L, interval=interval, nonlinear=F.relu, fullband=False, bn=True, time_emb_dim=time_dim)

        # Decoding
        for i in range(D - 1):
            L = L_in // (2 ** (D - 1 - i))
            in_ch_up = out_ch * 2
            out_ch //= ch_inc
            if verbose:
                print(f"Up {i+1}\t| C:{in_ch_up} -> {out_ch}\t| L:{L}")
            self.up.append(
                SHConvResnetBlock(Y, Y_inv, area, in_ch=in_ch_up, out_ch=out_ch, L=L, interval=interval, nonlinear=F.relu, fullband=False, bn=True, time_emb_dim=time_dim)
            )

        # final Up
        in_ch_up = out_ch * 2
        L *= 2
        if verbose:
            print(f"Up {D}\t| C:{in_ch_up} -> {out_ch}\t| L:{L}")
        self.up.append(
            SHConvResnetBlock(Y, Y_inv, area, in_ch=in_ch_up, out_ch=out_ch, L=L, interval=interval, nonlinear=F.relu, fullband=True, bn=True, time_emb_dim=time_dim)
        )

        # Final head
        if verbose:
            print(f"Final\t| C:{out_ch} -> {n_class}\t| L:{L_in}")
        self.final = SHConvBlock(Y, Y_inv, area, in_ch=out_ch, out_ch=n_class, L=L_in, interval=interval, nonlinear=None, fullband=True, bn=True)

    def forward(self, x, t=None, x_self_cond=None, time_obs=None, subject_id=None, ref_img=None):
        # self-conditioning
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat([x_self_cond, x], dim=1)  # (B, 2*in_ch, N)

        if self.add_xyz:
            B = x.shape[0]
            x = torch.cat([x, self.pos_xyz.expand(B, -1, -1)], dim=1)

        # time embedding
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        diff_time_emb = self.time_mlp(t)  # (B, 4C)

        cond_embs = [diff_time_emb] # 조건 임베딩 리스트

        # --- 2. Observation Time (T_obs) Embedding ---
        if self.max_time_obs is not None and time_obs is not None:
            norm_time_obs = time_obs / self.max_time_obs 
            time_obs_emb = self.time_obs_mlp(norm_time_obs)
            cond_embs.append(time_obs_emb)

        # --- 3. Subject ID (S_id) Embedding ---
        if self.num_subjects is not None and subject_id is not None:
            subject_emb = self.subject_emb(subject_id) # (B, 4C)
            cond_embs.append(subject_emb)

        # 🌟 [추가] 4. ViViT Reference Embedding ---
        if self.use_ref_condition and ref_img is not None:
            # ViViT forward (Input shape 확인 필요)
            ref_emb = self.vivit(ref_img) # (B, time_dim)
            ref_emb = self.vivit_proj(ref_emb)
            cond_embs.append(ref_emb)
        elif self.use_ref_condition and ref_img is None:
            # 학습 시나리오에 따라 Zero padding 혹은 에러 처리
            zero_emb = torch.zeros_like(diff_time_emb)
            cond_embs.append(zero_emb)

        # --- 5. Combine All Embeddings ---
        if len(cond_embs) > 1:
            combined_emb = torch.cat(cond_embs, dim=1) # (B, N * 4C)
            time_emb = self.combined_mlp(combined_emb) # (B, 4C)
        else:
            time_emb = diff_time_emb # (B, 4C)

        # Down
        feats = []
        y = x
        for blk in self.down:
            y = blk(y, time_emb=time_emb)
            feats.append(y)

        # Bottom
        y = self.bottom(y, time_emb=time_emb)

        # Up
        for i, blk in enumerate(self.up):
            skip = feats[-1 - i]
            y = torch.cat([y, skip], dim=1)
            y = blk(y, time_emb=time_emb)

        # Final
        y = self.final(y) # (B, n_class, N)
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
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        immiscible = False,
        # -------- geometry supervision (추가) --------
        faces = None,                    # (F,3) np/torch.long
        edge_face_indices = None,        # (Ne,2) np/torch.long, normal consistency용
        lambda_cyc: float = 0.0,         # 곡률 cycle loss 가중치
        curvature_loss: str = 'l1',      # 'l1' | 'l2'
        lambda_lap: float = 0.0,         # Laplacian smoothness 가중치
        lambda_normal: float = 0.0       # Normal consistency 가중치
        # -------------------------------------------
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.signal_size = signal_size

        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, \
            'objective must be pred_noise | pred_x0 | pred_v'

        # beta schedule
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

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper to register as float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # q(x_t | x_0) helpers
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # options
        self.immiscible = immiscible
        self.offset_noise_strength = offset_noise_strength

        # min-SNR loss weights
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

        # auto-normalization [0,1] <-> [-1,1]
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        # ---- geometry loss 설정 ----
        self.lambda_cyc   = float(lambda_cyc)
        self.curvature_loss = curvature_loss
        self.lambda_lap   = float(lambda_lap)
        self.lambda_normal= float(lambda_normal)

        if faces is not None:
            if isinstance(faces, np.ndarray):
                faces = torch.from_numpy(faces)
            self.register_buffer('faces', faces.to(dtype=torch.long))
        else:
            self.faces = None

        if edge_face_indices is not None:
            if isinstance(edge_face_indices, np.ndarray):
                edge_face_indices = torch.from_numpy(edge_face_indices)
            self.register_buffer('edge_face_indices', edge_face_indices.to(dtype=torch.long))
        else:
            self.edge_face_indices = None
        # --------------------------------

    @property
    def device(self):
        return self.betas.device

    # ---------- transforms between parameterizations ----------

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

    # ---------- posterior and model preds ----------

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, 
                          # 🌟 [추가] 인자
                          time_obs = None, subject_id = None, ref_img = None,
                          **kwargs):
        # 🌟 forward에 ref_img 전달
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
                        time_obs = None, subject_id = None, ref_img = None, # 🌟 [추가]
                        **kwargs):
        preds = self.model_predictions(x, t, x_self_cond, time_obs=time_obs, subject_id=subject_id, ref_img=ref_img, **kwargs)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start = x_start, x_t = x, t = t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # ---------- sampling ----------

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, 
                 time_obs = None, subject_id = None, ref_img = None, # 🌟 [추가]
                 **kwargs):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True, 
            time_obs=time_obs, subject_id=subject_id, ref_img=ref_img, # 🌟 전달
            **kwargs
        )
        noise = torch.randn_like(x) if t > 0 else 0
        pred = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False, 
                      time_obs = None, subject_id = None, ref_img = None, # 🌟 [추가]
                      **kwargs):
        batch, device = shape[0], self.device
        img = torch.randn(shape, device = device)
        imgs = [img]
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond, 
                                         time_obs=time_obs, subject_id=subject_id, ref_img=ref_img, # 🌟 전달
                                         **kwargs)
            imgs.append(img)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        return self.unnormalize(ret)

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps = False, 
                    time_obs = None, subject_id = None, ref_img = None, # 🌟 [추가]
                    **kwargs):
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
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, 
                time_obs=time_obs, subject_id=subject_id, ref_img=ref_img, # 🌟 전달
                **kwargs
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
               time_obs = None, subject_id = None, ref_img = None): # 🌟 ref_img 추가
        channels = self.channels
        N = sample_size if sample_size is not None else self.signal_size
        if N is None:
            raise ValueError("Please provide `sample_size` or set `signal_size` in GaussianDiffusion(...)")

        shape = (batch_size, channels, N)
        kwargs = dict(time_obs=time_obs, subject_id=subject_id, ref_img=ref_img)
        fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return fn(shape, return_all_timesteps = return_all_timesteps, **kwargs)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))
        img = (1 - lam) * xt1 + lam * xt2

        x_start = None
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None, 
                 time_obs = None, subject_id = None, ref_img = None): # 🌟 ref_img 추가
        b, c, n = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)
        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * offset_noise.unsqueeze(-1)

        # noisy sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # self-conditioning
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                # 🌟 self-cond 예측 시 ref_img 전달
                x_self_cond = self.model_predictions(x, t, time_obs=time_obs, subject_id=subject_id, ref_img=ref_img).pred_x_start
                x_self_cond.detach_()

        # predict & diffusion objective
        # 🌟 model forward에 ref_img 전달
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

        # ----- Geometry regularizers (옵션) -----
        # x0_hat 채널: [curv, x, y, z]
        if (self.faces is not None) and (x0_hat.shape[1] >= 4):
            curv_hat = x0_hat[:, 0:1, :]          # (B,1,N)
            xyz_hat  = x0_hat[:, 1:4, :]          # (B,3,N)
            verts = xyz_hat.transpose(1, 2).contiguous()  # (B,N,3)

            # 1) Cycle loss: mean curvature(verts) vs curv_hat
            if self.lambda_cyc > 0.0:
                _, H, _ = mean_curvature(verts, self.faces)  # H: (B,N)
                curv_pred = curv_hat.squeeze(1)              # (B,N)
                if self.curvature_loss.lower() == 'l2':
                    cyc = F.mse_loss(H, curv_pred, reduction='mean')
                else:
                    cyc = F.l1_loss(H, curv_pred, reduction='mean')
                loss = loss + self.lambda_cyc * cyc

            # 2) Laplacian smoothness (edge-aware Laplacian이면 더 좋음)
            if self.lambda_lap > 0.0:
                lap = lap_loss(verts, self.faces)             # (B,N)
                lap = lap.mean()
                loss = loss + self.lambda_lap * lap

            # 3) Normal consistency (인접 face 법선 유사)
            if (self.lambda_normal > 0.0) and (self.edge_face_indices is not None):
                nerr = normal_consistency(verts, self.faces, self.edge_face_indices)  # (B,Ne)
                nerr = nerr.mean()
                loss = loss + self.lambda_normal * nerr
        # ---------------------------------------

        return loss

    def forward(self, data_batch, *args, **kwargs):
        # data_batch가 (img, T_obs, S_id, Ref_img, names...) 튜플이라고 가정
        try:
            img = data_batch[0]
            time_obs = data_batch[1]
            subject_id = data_batch[2]
            
            # 🌟 [수정 핵심] 인덱스 3번에 있는 게 Tensor(이미지)인지 확인
            # 만약 Tensor가 아니라 리스트(names)라면 ref_img는 없는 것으로 처리
            if len(data_batch) > 3:
                candidate = data_batch[3]
                if isinstance(candidate, torch.Tensor):
                    ref_img = candidate
                else:
                    ref_img = None
            else:
                ref_img = None
            
            # device로 이동 (이미 Tensor라면 이동, 아니면 패스)
            if time_obs is not None: time_obs = time_obs.to(img.device)
            if subject_id is not None: subject_id = subject_id.to(img.device)
            if ref_img is not None: ref_img = ref_img.to(img.device)

        except (TypeError, IndexError):
            # (img,) 또는 (img, name)만 들어온 비-종단적 경우
            img = data_batch[0] if isinstance(data_batch, (tuple, list)) else data_batch
            time_obs = None
            subject_id = None
            ref_img = None

        b, device = img.shape[0], img.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = self.normalize(img)

        # 🌟 p_losses 호출 시 ref_img 전달
        return self.p_losses(img, t, *args, 
                             time_obs=time_obs, 
                             subject_id=subject_id,
                             ref_img=ref_img,
                             **kwargs)


class SphericalTrainer:
    def __init__(
        self, diffusion_model, dataset, *, train_batch_size = 16, gradient_accumulate_every = 1, train_lr = 1e-4, train_num_steps = 100_000, ema_update_every = 10, ema_decay = 0.995,
        adam_betas = (0.9, 0.99), save_and_sample_every = 1000, num_samples = 16, results_folder = './results_sph', mixed_precision_type = 'fp16', split_batches = True,
        max_grad_norm = 1.0, save_best_and_latest_only = False, sphere_path: str = None, log_dir = './logs', valid_dataset = None
    ):
        super().__init__()

        # self.device = diffusion_model.device
        self.model = diffusion_model.to(diffusion_model.device)
        self.channels = self.model.channels
        self.is_ddim_sampling = self.model.is_ddim_sampling

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 1, \
            'effective batch size (batch_size x grad_accum) 는 16 이상 권장'

        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        # dataset / dataloader
        self.ds = dataset
        self.dl = DataLoader(
            self.ds,
            batch_size = train_batch_size,
            shuffle = True,
            # pin_memory = True,
            # num_workers = cpu_count(),
            collate_fn = spherical_collate_longitudinal
        )
        self.data_iter = self._cycle(self.dl)
        
        # 검증 데이터로더 생성
        self.valid_dl = None
        if valid_dataset is not None:
            self.valid_dl = DataLoader(
                valid_dataset,
                batch_size = train_batch_size, # (훈련 배치 크기와 동일하게 사용)
                shuffle = False, # (검증셋은 섞지 않음)
                collate_fn = spherical_collate_longitudinal
            )
            print(f"검증 데이터셋 로드 완료. {len(valid_dataset)}개 샘플.")
            
            
            

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
        if self.save_best_and_latest_only:
            print("`save_best_and_latest_only=True`이지만 FID가 없어 best를 판단할 기준이 없습니다. 일반 저장 모드를 권장합니다.")

        self.sphere_path = sphere_path


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

        print(f"Loaded checkpoint at step {self.step}")


    def train(self):
        device = self.model.device
        
        with tqdm(initial = self.step, total = self.train_num_steps, disable = False) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                total_loss = 0.0
                
                # 1. 배치 ID를 저장할 리스트 초기화
                current_batch_ids = [] 

                for _ in range(self.gradient_accumulate_every):
                    # 2. 배치를 가져옴: (x, T_obs, S_id, Ref_img, names) 구조 예상
                    data_batch = next(self.data_iter)
                    
                    # 3. names (MRI ID 리스트) 추출 (names는 보통 튜플의 마지막)
                    current_names = data_batch[-1] if isinstance(data_batch[-1], list) else []
                    current_batch_ids.extend(current_names)
                    
                    # 4. 텐서만 GPU로 이동
                    # data_batch는 튜플. 텐서인 요소들만 to(device) 처리
                    data_batch_on_device = []
                    for item in data_batch:
                        if isinstance(item, torch.Tensor):
                            data_batch_on_device.append(item.to(device))
                        else:
                            data_batch_on_device.append(item)
                    
                    data_batch_on_device = tuple(data_batch_on_device)

                    with autocast(enabled=False):
                        # forward 내부에서 ref_img 자동 처리
                        loss = self.model(data_batch_on_device)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    loss.backward()
                    
                # 5. 그래디언트 누적 후 출력
                processed_ids = ", ".join(current_batch_ids[:5]) + ("..." if len(current_batch_ids)>5 else "")
                pbar.set_description(f'Loss: {total_loss:.4f} | Data: [{processed_ids}]')

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()
                
                self.loss_history.append(total_loss)
                
                # TensorBoard에 기록
                if hasattr(self, 'writer'):
                    self.writer.add_scalar('Loss/train', total_loss, self.step)

                self.step += 1
                if hasattr(self, 'ema'):
                    self.ema.update()

                # --- 그래프 및 샘플 저장 ---
                if self.step != 0 and (self.step % self.save_and_sample_every == 0):
                    self._sample_and_save()
                    
                    # 체크포인트 저장
                    milestone = self.step // self.save_and_sample_every
                    if self.save_best_and_latest_only:
                        self.save("latest")
                    else:
                        self.save(milestone)
                        
                    # --- 실시간 Matplotlib 그래프 저장 ---
                    if len(self.loss_history) > 0:
                        print(f'\nStep {self.step}: Saving real-time loss graph...')
                        try:
                            plt.figure(figsize=(12, 6))
                            plt.plot(self.loss_history)
                            plt.title(f'Training Loss (up to Step {self.step})')
                            plt.xlabel('Step')
                            plt.ylabel('Loss')
                            plt.grid(True)
                            
                            # 실시간 그래프 덮어쓰기
                            graph_path = self.results_folder / 'loss_graph_realtime.png'
                            plt.savefig(graph_path)
                            plt.close() 
                            print(f"Real-time loss graph saved to {graph_path}")
                        except Exception as e:
                            print(f"Failed to update loss graph: {e}")
                            
                pbar.update(1)

        print('training complete')

        # --- 최종 그래프 저장 ---
        if len(self.loss_history) > 0:
            print('Saving final loss graph...')
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(self.loss_history)
                plt.title('Training Loss Over Steps (Final)')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.grid(True)
                
                graph_path = self.results_folder / 'loss_graph_final.png'
                plt.savefig(graph_path)
                plt.close()
                print(f"Final loss graph saved to {graph_path}")
            except Exception as e:
                print(f"Failed to save final loss graph: {e}")
                
                
    
    @torch.no_grad()
    def _sample_and_save(self):
        if not hasattr(self, 'ema'):
            return

        self.ema.ema_model.eval()

        try:
            # 데이터셋에서 정점(vertex) 개수 N을 가져옵니다.
            N = getattr(self.ds, 'num_vertices') # OASISLongitudinalDataset에서 정의된 속성 사용
        except Exception:
            N = getattr(self.ema.ema_model, 'signal_size', None)

        milestone = self.step // self.save_and_sample_every
        batches = num_to_groups(self.num_samples, self.batch_size)
        
        # 🌟 [수정 1: 샘플링 조건 생성]
        B = self.num_samples # 총 샘플 수
        device = self.device
        
        # 1. Subject ID (피험자 ID) 조건 생성
        total_subjects = self.ds.num_subjects if hasattr(self.ds, 'num_subjects') else 0
        if total_subjects > 0:
            subject_ids_int = torch.randint(0, total_subjects, (B,), device=device).long()
        else:
            subject_ids_int = None
        
        # 2. MR Delay (관찰 시간 T_obs) 조건 생성
        max_t = self.ds.max_time_obs if hasattr(self.ds, 'max_time_obs') else 0
        if max_t is not None and max_t > 0:
            time_obs_float = torch.rand(B, device=device) * max_t
        else:
            time_obs_float = None
            
        # 🌟 [추가] 3. Reference Mesh (ViViT용) 조건 생성
        # 학습된 ViViT가 유의미하게 작동하려면 실제 데이터 분포와 유사한 ref_img를 넣어줘야 함
        ref_img_sample = None
        if self.model.model.use_ref_condition:
             # self.model.model.vivit의 input shape에 맞춰야 함
             # 예: (B, C, Frames, Vertices) or (B, C, Vertices)
             ref_channels = self.model.model.vivit.channels if hasattr(self.model.model.vivit, 'channels') else 1
             ref_frames = self.model.model.vivit.frames if hasattr(self.model.model.vivit, 'frames') else 1
             
             # 여기서는 랜덤 노이즈로 예시 (실제 사용 시에는 검증셋에서 추출 권장)
             if ref_frames > 1:
                 ref_img_sample = torch.randn(B, ref_channels, ref_frames, N, device=device)
             else:
                 # vivit 구현에 따라 차원 다를 수 있음
                 ref_img_sample = torch.randn(B, ref_channels, N, device=device) 
        
        # 출력 메시지
        print(f"\n--- Sampling Start (Milestone {milestone}) ---")
        print(f"Total {B} samples to generate.")
        
        # 🌟 [수정 2: 샘플링 시 조건 전달]
        all_samples = []
        start_idx = 0
        for n in batches:
            # 현재 배치 크기에 맞는 조건 슬라이싱
            current_s_id = subject_ids_int[start_idx:start_idx + n] if subject_ids_int is not None else None
            current_t_obs = time_obs_float[start_idx:start_idx + n] if time_obs_float is not None else None
            
            # 🌟 [중요] ref_img 슬라이싱
            current_ref = ref_img_sample[start_idx:start_idx + n] if ref_img_sample is not None else None
            
            samp = self.ema.ema_model.sample(
                batch_size=n, 
                sample_size=N, 
                return_all_timesteps=False,
                time_obs=current_t_obs,       # 👈 Time 조건
                subject_id=current_s_id,      # 👈 Subject ID 조건
                ref_img=current_ref           # 🌟 Reference Mesh 조건
            )
            all_samples.append(samp)
            start_idx += n
            
        samples = torch.cat(all_samples, dim=0)

        # 샘플 저장
        npy_path = self.results_folder / f'samples-{milestone}.npy'
        np.save(npy_path, samples.detach().cpu().numpy())
        print(f"--- Samples Saved to {npy_path} ---")
        
        # 🌟 조건(Ref image) 저장 (분석용)
        if ref_img_sample is not None:
            ref_npy_path = self.results_folder / f'condition_ref-{milestone}.npy'
            np.save(ref_npy_path, ref_img_sample.detach().cpu().numpy())
            print(f"--- Condition Ref Image Saved to {ref_npy_path} ---")
