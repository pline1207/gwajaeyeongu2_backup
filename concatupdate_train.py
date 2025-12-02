import os
import argparse
import numpy as np
from tqdm import tqdm
from contextlib import ExitStack
import nibabel as nib
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd 

# spharmnet 임포트
from spharmnet import SPHARM_Net, SphericalTrainer, GaussianDiffusion
from spharmnet.lib.io import read_mesh, read_dat


class OASISLongitudinalDataset(Dataset):
    def __init__(self, csv_path, surf_base_dir, in_ch, hemi, 
                 subject_to_id_map: dict,
                 subject_partition_set: set,
                 max_time_obs_overall: float, 
                 data_norm=False, preload="none", data_normalization=False, 
                 sphere_path=None, 
                 partition='train',
                 p_uncond=0.0 # 🌟 [CFG] Unconditional 학습 확률
                 ):
        super().__init__()
        
        self.surf_base_dir = os.path.join(surf_base_dir, "features") 
        self.in_ch = in_ch
        self.hemi = hemi
        self.preload = preload
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.partition = partition
        self.p_uncond = p_uncond

        if sphere_path is None:
            raise ValueError("OASISLongitudinalDataset requires 'sphere_path'.")
        
        ico_v, _ = read_mesh(sphere_path) 
        self.num_vertices = ico_v.shape[0]

        # 1. Excel 로드
        full_df = pd.read_excel(csv_path) 
        full_df = full_df.dropna(subset=['Subject ID'])
        
        # Age 컬럼 필수 확인
        if 'Age' not in full_df.columns:
            raise ValueError("Excel 파일에 'Age' 컬럼이 있어야 합니다.")

        # Partitioning
        self.df = full_df[full_df['Subject ID'].isin(subject_partition_set)].reset_index(drop=True)
        
        self.subject_to_id = subject_to_id_map
        self.num_subjects = len(subject_to_id_map) 
        self.max_time_obs = max_time_obs_overall 

        print(f"[{partition.upper()}] Loaded {len(self.df)} scans. (CFG p_uncond={p_uncond})")

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
                    
                    # 정규화
                    if 'thickness' in ch:
                        surface_data = np.clip(surface_data, 0, 5.0) / 5.0
                    elif 'curv' in ch or 'sulc' in ch or 'K1' in ch or 'K2' in ch:
                        surface_data = (np.clip(surface_data, -1.0, 1.0) + 1.0) / 2.0
                    else:
                        surface_data = np.clip(surface_data, 0.0, 1.0)

                    if surface_data.ndim == 1:
                        surface_data = surface_data[np.newaxis, :] 
                        
                    all_channels_data.append(surface_data)
                    
                except Exception as e:
                    print(f"Error loading: {file_path}") 
                    raise e

        data = np.concatenate(all_channels_data, axis=0) 
        return data.astype(np.float32)

    def __getitem__(self, idx):
        # 1. Target 데이터 로드
        row = self.df.iloc[idx]
        subject_id_str = row['Subject ID']
        mri_id = row['MRI ID']
        age = float(row['Age']) / 100.0 
        subject_id_int = int(self.subject_to_id[subject_id_str])
        
        target_data = self._load_scan_data(mri_id)

        # ==========================================================
        # 🌟 [Cheating Mode] 무조건 정답(Target)을 Reference로 사용
        # ==========================================================
        needed_frames = 2
        
        # 실제 Reference 로직을 무시하고, Target 데이터를 복제해서 사용합니다.
        # 이렇게 하면 모델은 정답을 미리 보고 베끼는 연습을 하게 됩니다.
        ref_data = np.stack([target_data] * needed_frames, axis=0)
        
        # ==========================================================
        # 🌟 [CFG Training] 확률적으로 Reference 지우기
        # ==========================================================
        # Cheating 모드라 할지라도, 가중치 조절(Guidance)을 하려면
        # 모델이 "Reference가 없는 상황"도 학습해야 합니다.
        if self.partition == 'train' and self.p_uncond > 0:
            if np.random.rand() < self.p_uncond:
                ref_data = np.zeros_like(ref_data)

        # 4. 리턴: (Target, MRI_ID, Age, ID, Reference)
        return target_data, mri_id, age, subject_id_int, ref_data


def get_args():
    parser = argparse.ArgumentParser()

    # Dataset & dataloader
    parser.add_argument("--sphere", type=str, default="/data/object/sphere/unist/icosphere_6.vtk", help="Sphere mesh (vtk or FreeSurfer format)")
    parser.add_argument("--data-norm", action="store_true", help="Z-score+prctile data normalization")
    parser.add_argument("--preload", type=str, choices=["none", "cpu", "device"], default="device", help="Data preloading")
    parser.add_argument("--in-ch", type=str, default=["curv", "sulc", "inflated.H"], nargs="+", help="List of geometry")
    parser.add_argument("--hemi", type=str, nargs="+", choices=["lh", "rh"], help="Hemisphere for learning", required=True)
    
    parser.add_argument("--csv-file", type=str, default="/data/human/OASIS/OASIS2/oasis_longitudinal_demographics.xlsx", required=True, help="OASIS longitudinal Excel 파일 경로")
    parser.add_argument("--surf-dir", type=str, default="/data/human/OASIS/OASIS2/Freesurfer", required=True, help="Freesurfer 표면 데이터가 있는 기본 디렉토리")
    
    parser.add_argument("--classes", type=int, nargs="+", help="List of regions of interest")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data shuffling")
    parser.add_argument("--aug", type=int, default=0, help="Level of data augmentation")

    parser.add_argument("--test-split-ratio", type=float, default=0.2,help="테스트셋으로 분리할 피험자 비율")
    
    # Training and evaluation
    parser.add_argument("--train-num-steps", type=int, default=100000, help="Max epoch")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=8e-5, help="Initial learning rate")
    parser.add_argument("--no-decay", action="store_true", help="Disable decay (every 2 epochs if no progress)")
    parser.add_argument("--loss", type=str, default="dl", choices=["dl", "ce"], help="dl: Dice loss, ce: cross entropy")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Path to the log files (output)")
    parser.add_argument("--ckpt-dir", type=str, default="./logs", help="Path to the checkpoint file (output)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint (pth) to resume training")
    
    # 🌟 [수정] 그래프/샘플링 빈도 완화 (기본 2000 step마다 저장)
    parser.add_argument("--save-and-sample-every", type=int, default=2000, help="너무 빽빽하지 않게 저장 빈도 조절")
    
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--results-dir", type=str, default="/data/lfs/pline1207/SPHARM-Diffusion/results")
    parser.add_argument("--data-normalization", action="store_true")
    
    # diffusion settings
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-timesteps", type=int, default=None)
    parser.add_argument("--objective", type=str, choices=["pred_v", "pred_x0", "pred_noise"], default="pred_v")
    parser.add_argument("--beta-schedule", type=str, choices=["linear", "cosine", "sigmoid"], default="cosine")
    parser.add_argument("--auto-normalize", action="store_true")
    
    # SPHARM-Net settings
    parser.add_argument("-D", "--depth", type=int, default=3, help="Depth of SPHARM-Net")
    parser.add_argument("-C", "--channel", type=int, default=128, help="# of channels in the entry layer of SPHARM-Net")
    parser.add_argument("-L", "--bandwidth", type=int, default=80, help="Bandwidth of SPHARM-Net")
    parser.add_argument("--interval", type=int, default=5, help="Anchor interval of hamonic coefficients")
    
    # ViViT & CFG Settings
    parser.add_argument("--use-ref", action="store_true", help="ViViT를 사용해 Reference Image를 조건으로 줄 것인지 여부")
    parser.add_argument("--ref-frames", type=int, default=2, help="ViViT에 들어갈 Reference Frame 수 (보통 2)")
    # 🌟 [수정] CFG 학습을 위한 p-uncond 기본값 설정 (0.2 권장)
    parser.add_argument("--p-uncond", type=float, default=0.2, help="학습 중 Reference를 0으로 만들 확률 (CFG용)")

    # Machine settings
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for training")
    parser.add_argument("--no-cuda", action="store_true", help="No CUDA")
    parser.add_argument("--threads", type=int, default=1, help="# of CPU threads")

    args = parser.parse_args()
    return args


def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    preload = None if args.preload == "none" else device if args.preload == "device" else args.preload

    torch.set_num_threads(args.threads)
    if not args.cuda:
        torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("Loading data...")
    sphere = os.path.join(args.sphere)
    v, _ = read_mesh(sphere)

    # 통합 ID 맵 생성
    print(f"사전 로드 (통합 ID 맵 생성용): {args.csv_file}")
    try:
        full_df = pd.read_excel(args.csv_file)
        full_df = full_df.dropna(subset=['Subject ID'])
    except Exception as e:
        raise FileNotFoundError(f"Excel 파일 로드 실패: {args.csv_file} - {e}")

    all_subjects_sorted = sorted(full_df['Subject ID'].unique())
    total_unique_subjects = len(all_subjects_sorted)
    
    subject_to_id_map = {subj: i for i, subj in enumerate(all_subjects_sorted)}
    
    print(f"총 {total_unique_subjects}명의 고유 피험자에 대해 통합 ID 맵 생성 완료.")

    # 훈련/테스트 분리
    rng = np.random.RandomState(args.seed)
    shuffled_subjects = list(all_subjects_sorted)
    rng.shuffle(shuffled_subjects)
    
    split_idx = int(total_unique_subjects * (1.0 - args.test_split_ratio))
    train_subjects_set = set(shuffled_subjects[:split_idx])
    test_subjects_set = set(shuffled_subjects[split_idx:])
    
    # max_time_obs 계산
    full_df['MR Delay'] = full_df['MR Delay'].apply(lambda x: str(x).replace('M', '').strip())
    full_df['MR Delay'] = pd.to_numeric(full_df['MR Delay'], errors='coerce').fillna(0)
    max_time_obs = full_df['MR Delay'].max()

    # 훈련 데이터셋 (p_uncond 적용하여 CFG 학습 수행)
    ds_train = OASISLongitudinalDataset(
        csv_path=args.csv_file,
        surf_base_dir=args.surf_dir,
        in_ch=args.in_ch,
        hemi=args.hemi,
        subject_to_id_map=subject_to_id_map,
        subject_partition_set=train_subjects_set,
        max_time_obs_overall=max_time_obs,
        data_norm=args.data_norm,
        preload=preload,
        data_normalization=args.data_normalization,
        sphere_path=sphere,
        partition='train',
        p_uncond=args.p_uncond # CFG 확률 전달
    )

    # 테스트 데이터셋 (테스트는 p_uncond=0으로 하여 항상 Cheating/Conditioning 모드)
    ds_test = OASISLongitudinalDataset(
        csv_path=args.csv_file,
        surf_base_dir=args.surf_dir,
        in_ch=args.in_ch,
        hemi=args.hemi,
        subject_to_id_map=subject_to_id_map,
        subject_partition_set=test_subjects_set,
        max_time_obs_overall=max_time_obs,
        data_norm=args.data_norm,
        preload="none", 
        data_normalization=args.data_normalization,
        sphere_path=sphere,
        partition='test',
        p_uncond=0.0 # 테스트는 항상 조건 유지
    )

    model_in_channels = len(args.in_ch) * len(args.hemi)
    if model_in_channels == 0:
        raise ValueError("입력 채널(--in-ch)과 반구(--hemi)를 1개 이상 지정해야 합니다.")

    print(f"모델에 전달할 총 고유 피험자 수: {total_unique_subjects}")
    print(f"ViViT Reference 사용 여부: {args.use_ref}")
    print(f"CFG Unconditional Prob: {args.p_uncond}")
    
    # SPHARM_Net 호출
    model = SPHARM_Net(
        sphere=sphere,
        device=device,
        in_ch=model_in_channels,
        n_class=model_in_channels,
        C=args.channel,
        L=args.bandwidth,
        D=args.depth,
        interval=args.interval,
        self_condition=False,
        verbose=False,
        add_xyz=True,
        max_time_obs=max_time_obs,
        num_subjects=total_unique_subjects,
        # ViViT 활성화
        use_ref_condition=args.use_ref,      
        ref_in_ch=model_in_channels,        
        ref_num_frames=args.ref_frames      
    )
    model.to(device)

    print("train: auto_normalize: ", args.auto_normalize)
    diffusion = GaussianDiffusion(
        model=model,
        signal_size=v.shape[0],
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective=args.objective,
        beta_schedule=args.beta_schedule,
        auto_normalize=args.auto_normalize
    )
    diffusion.to(device)
    
    trainer = SphericalTrainer(
        diffusion_model=diffusion,
        dataset=ds_train,
        train_batch_size=args.batch_size,
        gradient_accumulate_every=2,
        train_lr=args.learning_rate,
        train_num_steps=args.train_num_steps,
        ema_decay=0.995,
        results_folder=args.results_dir,
        # 🌟 [수정] 너무 잦은 저장을 막기 위해 Argument에서 설정된 값 사용
        save_and_sample_every=args.save_and_sample_every, 
        num_samples=args.num_samples,
        log_dir=args.log_dir,
        valid_dataset=ds_test 
    )
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    main(args)
