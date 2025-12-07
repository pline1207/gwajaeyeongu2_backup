import os
import argparse
import numpy as np
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd 

# new_model.py에서 필요한 모듈 임포트
# [주의] new_model.py 파일이 spharmnet 폴더 안에 있거나, 동일 디렉토리에 있어야 합니다.
from spharmnet.core.new_model import SPHARM_Net, SphericalTrainer, GaussianDiffusion, spherical_collate_longitudinal
from spharmnet.lib.io import read_mesh, read_dat

# --------------------------------------------------------------------------
# 1. HCP Dataset (Pre-training)
#    Path: /data/prep/stable/HCP/features
#    Format: {subject_id}.{hemi}.aug0.{feature}.dat
# --------------------------------------------------------------------------
class HCPDataset(Dataset):
    def __init__(self, surf_base_dir, in_ch, hemi, sphere_path, p_uncond=0.2):
        super().__init__()
        self.surf_base_dir = surf_base_dir
        self.in_ch = in_ch
        self.hemi = hemi
        self.p_uncond = p_uncond # CFG null probability

        if sphere_path is None: raise ValueError("Sphere path required")
        ico_v, _ = read_mesh(sphere_path) 
        self.num_vertices = ico_v.shape[0]

        # 파일 스캔 및 Subject ID 추출
        print(f"Scanning HCP directory: {surf_base_dir} ...")
        # 예: 100206.lh.aug0.sulc.dat -> 100206 추출
        all_files = glob.glob(os.path.join(surf_base_dir, f"*.{hemi[0]}.aug0.{in_ch[0]}.dat"))
        
        self.subject_ids = []
        for f in all_files:
            fname = os.path.basename(f)
            sid = fname.split('.')[0] # "100206"
            self.subject_ids.append(sid)
        
        self.subject_ids = sorted(list(set(self.subject_ids)))
        self.num_subjects = len(self.subject_ids)
        self.subject_to_idx = {sid: i for i, sid in enumerate(self.subject_ids)}
        
        print(f"[HCP] Found {self.num_subjects} subjects.")

    def __len__(self):
        return len(self.subject_ids)

    def _load_data(self, sid):
        all_channels_data = []
        for h in self.hemi:     
            for ch in self.in_ch: 
                # HCP 파일명 포맷: {id}.{hemi}.aug0.{feat}.dat
                file_name = f"{sid}.{h}.aug0.{ch}.dat"
                file_path = os.path.join(self.surf_base_dir, file_name)
                try:
                    surface_data = read_dat(file_path, self.num_vertices)
                    # Normalization
                    if 'thickness' in ch: surface_data = np.clip(surface_data, 0, 5.0) / 5.0
                    elif 'curv' in ch or 'sulc' in ch: surface_data = (np.clip(surface_data, -1.0, 1.0) + 1.0) / 2.0
                    else: surface_data = np.clip(surface_data, 0.0, 1.0)
                    
                    if surface_data.ndim == 1: surface_data = surface_data[np.newaxis, :] 
                    all_channels_data.append(surface_data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    return None
        return np.concatenate(all_channels_data, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        data = self._load_data(sid)
        
        if data is None: # 에러시 임시 처리
             data = np.zeros((len(self.hemi)*len(self.in_ch), self.num_vertices), dtype=np.float32)

        # HCP는 단일 시점이므로 time_obs = 0
        # Pre-training에서 ViViT 학습을 위해 '자기 자신'을 Reference로 줌 (Reconstruction Task)
        ref_data = np.stack([data] * 2, axis=0) # (Frames, C, N)
        
        # CFG Training
        if np.random.rand() < self.p_uncond:
            ref_data = np.zeros_like(ref_data)

        # Output format: (Target, Name, Time, SubjID, Ref)
        return data, sid, 0.0, self.subject_to_idx[sid], ref_data

# --------------------------------------------------------------------------
# 2. OASIS Dataset (Fine-tuning)
# --------------------------------------------------------------------------
class OASISLongitudinalDataset(Dataset):
    def __init__(self, csv_path, surf_base_dir, in_ch, hemi, 
                 subject_to_id_map: dict,
                 subject_partition_set: set,
                 max_time_obs_overall: float, 
                 data_norm=False, preload="none", data_normalization=False, 
                 sphere_path=None, 
                 partition='train',
                 p_uncond=0.0
                 ):
        super().__init__()
        
        self.surf_base_dir = os.path.join(surf_base_dir, "features") 
        self.in_ch = in_ch
        self.hemi = hemi
        self.partition = partition
        self.p_uncond = p_uncond

        if sphere_path is None: raise ValueError("sphere_path required.")
        ico_v, _ = read_mesh(sphere_path) 
        self.num_vertices = ico_v.shape[0]

        # Excel Load
        full_df = pd.read_excel(csv_path) 
        full_df = full_df.dropna(subset=['Subject ID'])
        self.df = full_df[full_df['Subject ID'].isin(subject_partition_set)].reset_index(drop=True)
        self.subject_to_id = subject_to_id_map
        self.num_subjects = len(subject_to_id_map) 
        self.max_time_obs = max_time_obs_overall 
        print(f"[OASIS-{partition.upper()}] Loaded {len(self.df)} scans.")

    def __len__(self):
        return len(self.df)

    def _load_scan_data(self, mri_id):
        all_channels_data = []
        for h in self.hemi:     
            for ch in self.in_ch: 
                file_name = f"{mri_id}.{h}.aug0.{ch}.dat"
                file_path = os.path.join(self.surf_base_dir, file_name)
                try:
                    surface_data = read_dat(file_path, self.num_vertices)
                    if 'thickness' in ch: surface_data = np.clip(surface_data, 0, 5.0) / 5.0
                    elif 'curv' in ch or 'sulc' in ch: surface_data = (np.clip(surface_data, -1.0, 1.0) + 1.0) / 2.0
                    else: surface_data = np.clip(surface_data, 0.0, 1.0)
                    if surface_data.ndim == 1: surface_data = surface_data[np.newaxis, :] 
                    all_channels_data.append(surface_data)
                except Exception as e:
                    raise e
        return np.concatenate(all_channels_data, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id_str = row['Subject ID']
        mri_id = row['MRI ID']
        age = float(row['Age']) / 100.0 
        subject_id_int = int(self.subject_to_id[subject_id_str])
        target_data = self._load_scan_data(mri_id)

        # OASIS: 실제 이전 시점 혹은 기준 시점을 Reference로 사용 (여기서는 Simplified: 자기 자신 복제)
        # 실제 연구에서는 이 부분을 '이전 방문 데이터'를 로드하도록 수정해야 함
        ref_data = np.stack([target_data] * 2, axis=0) 
        
        if self.partition == 'train' and self.p_uncond > 0:
            if np.random.rand() < self.p_uncond:
                ref_data = np.zeros_like(ref_data)

        return target_data, mri_id, age, subject_id_int, ref_data


def get_args():
    parser = argparse.ArgumentParser()
    # Mode Selection
    parser.add_argument("--dataset", type=str, choices=["HCP", "OASIS"], required=True, help="Choose dataset for Pre-training (HCP) or Fine-tuning (OASIS)")
    parser.add_argument("--pretrained-weights", type=str, default=None, help="Path to pre-trained model .pt file")

    # Dataset Paths
    parser.add_argument("--sphere", type=str, default="/data/object/sphere/unist/icosphere_6.vtk")
    parser.add_argument("--hcp-dir", type=str, default="/data/prep/stable/HCP/features")
    parser.add_argument("--oasis-csv", type=str, default="/data/human/OASIS/OASIS2/oasis_longitudinal_demographics.xlsx")
    parser.add_argument("--oasis-dir", type=str, default="/data/human/OASIS/OASIS2/Freesurfer")
    
    # Common Args
    parser.add_argument("--in-ch", type=str, default=["curv", "sulc", "inflated.H"], nargs="+")
    parser.add_argument("--hemi", type=str, nargs="+", choices=["lh", "rh"], default=["lh", "rh"])
    
    # Training
    parser.add_argument("--train-num-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--log-dir", type=str, default="./logs")
    
    # Model
    parser.add_argument("--channel", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--bandwidth", type=int, default=80)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--use-ref", action="store_true", default=True) # Always use Ref (Self for HCP)
    
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    
    return parser.parse_args()


def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    
    sphere = args.sphere
    v, _ = read_mesh(sphere)

    # 1. Dataset Initialization
    if args.dataset == "HCP":
        print(">>> Initializing HCP Dataset (Pre-training Mode)")
        ds_train = HCPDataset(surf_base_dir=args.hcp_dir, in_ch=args.in_ch, hemi=args.hemi, sphere_path=sphere, p_uncond=0.2)
        ds_test = None 
        
        num_subjects = ds_train.num_subjects
        max_time_obs = 1.0 # 의미 없음
        results_folder = os.path.join(args.results_dir, "HCP_Pretrain")

    elif args.dataset == "OASIS":
        print(">>> Initializing OASIS Dataset (Fine-tuning Mode)")
        full_df = pd.read_excel(args.oasis_csv).dropna(subset=['Subject ID'])
        all_subjects = sorted(full_df['Subject ID'].unique())
        subj_map = {s: i for i, s in enumerate(all_subjects)}
        
        # Split
        split_idx = int(len(all_subjects) * 0.8)
        train_set = set(all_subjects[:split_idx])
        test_set = set(all_subjects[split_idx:])
        
        # Delay Max
        full_df['MR Delay'] = pd.to_numeric(full_df['MR Delay'].apply(lambda x: str(x).replace('M', '').strip()), errors='coerce').fillna(0)
        max_time_obs = full_df['MR Delay'].max()

        ds_train = OASISLongitudinalDataset(args.oasis_csv, args.oasis_dir, args.in_ch, args.hemi, subj_map, train_set, max_time_obs, sphere_path=sphere, partition='train', p_uncond=0.2)
        ds_test = OASISLongitudinalDataset(args.oasis_csv, args.oasis_dir, args.in_ch, args.hemi, subj_map, test_set, max_time_obs, sphere_path=sphere, partition='test', p_uncond=0.0)
        
        num_subjects = len(all_subjects)
        results_folder = os.path.join(args.results_dir, "OASIS_Finetune")
    
    # 2. Model Setup
    model_in_channels = len(args.in_ch) * len(args.hemi)
    
    model = SPHARM_Net(
        sphere=sphere, device=device, in_ch=model_in_channels, n_class=model_in_channels,
        C=args.channel, L=args.bandwidth, D=args.depth, 
        max_time_obs=max_time_obs if args.dataset == "OASIS" else None, # HCP는 Time Emb 안씀
        num_subjects=num_subjects,
        use_ref_condition=args.use_ref, ref_in_ch=model_in_channels, ref_num_frames=2
    )
    model.to(device)

    # 3. Load Pretrained Weights (If provided)
    if args.pretrained_weights:
        print(f"\n[Transfer Learning] Loading weights from {args.pretrained_weights}")
        model.load_pretrained_weights(args.pretrained_weights, device)

    # 4. Diffusion Setup
    diffusion = GaussianDiffusion(
        model=model, signal_size=v.shape[0], timesteps=args.timesteps,
        beta_schedule='cosine', objective='pred_v'
    )
    diffusion.to(device)

    # 5. Trainer
    trainer = SphericalTrainer(
        diffusion_model=diffusion, dataset=ds_train, train_batch_size=args.batch_size,
        train_lr=args.learning_rate, train_num_steps=args.train_num_steps,
        results_folder=results_folder, valid_dataset=ds_test,
        save_and_sample_every=2000
    )
    
    print(f"\n>>> Start Training: {args.dataset} Mode")
    trainer.train()

if __name__ == "__main__":
    args = get_args()
    main(args)
