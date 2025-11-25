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
                 max_time_obs_overall: float, # 여기서는 max_age로 쓰거나 무시 가능
                 data_norm=False, preload="none", data_normalization=False, 
                 sphere_path=None, 
                 partition='train'
                 ):
        super().__init__()
        
        self.surf_base_dir = os.path.join(surf_base_dir, "features") 
        self.in_ch = in_ch
        self.hemi = hemi
        self.preload = preload
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.partition = partition

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
        self.max_time_obs = max_time_obs_overall # 사용 안 할 수도 있음 (Age/100 사용시)

        print(f"[{partition.upper()}] Loaded {len(self.df)} scans.")

    def __len__(self):
        return len(self.df)

    # 🌟 [Helper 함수] 파일 로드 및 정규화 로직 분리 (중복 제거)
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
                    # 에러 발생 시 0으로 채운 더미 데이터 반환 (학습이 죽지 않도록) 혹은 raise
                    raise e

        data = np.concatenate(all_channels_data, axis=0) 
        return data.astype(np.float32)

    def __getitem__(self, idx):
        # 1. 현재(Target) 샘플 정보
        row = self.df.iloc[idx]
        subject_id_str = row['Subject ID']
        mri_id = row['MRI ID']
        
        # 🌟 Age 정규화 (0~1 범위로 맞춤, 예를 들어 100세 기준)
        age = float(row['Age']) / 100.0 
        
        subject_id_int = int(self.subject_to_id[subject_id_str])
        
        # 2. Target 데이터 로드
        target_data = self._load_scan_data(mri_id)

        # 3. 🌟 [핵심] Reference 데이터 찾기 (같은 사람의 다른 데이터)
        # 현재 데이터프레임에서 같은 ID를 가진 행들을 찾음
        subj_rows = self.df[self.df['Subject ID'] == subject_id_str]
        
        # 현재 스캔(idx)을 제외한 나머지 후보군
        # (주의: iloc index와 df index가 다를 수 있으므로 MRI ID로 비교 추천)
        candidates = subj_rows[subj_rows['MRI ID'] != mri_id]
        
        if len(candidates) > 0:
            # 다른 시점의 데이터가 있다면 랜덤하게 하나 선택 (Data Augmentation 효과)
            ref_row = candidates.sample(n=1).iloc[0]
            ref_mri_id = ref_row['MRI ID']
            ref_data = self._load_scan_data(ref_mri_id)
        else:
            # 만약 스캔이 하나뿐인 사람이라면? 자기 자신을 Reference로 사용 (혹은 Zero tensor)
            # 여기서는 자기 자신을 사용하는 것이 코드 안정성에 좋음
            ref_data = target_data.copy()

        # 4. 리턴: (Target, MRI_ID, Age, ID, Reference)
        # new_model.py의 collate_fn 순서에 맞춤: data, name, time, id, ref
        return target_data, mri_id, age, subject_id_int, ref_data


def get_args():
    parser = argparse.ArgumentParser()

    # Dataset & dataloader
    parser.add_argument("--sphere", type=str, default="/data/object/sphere/unist/icosphere_6.vtk", help="Sphere mesh (vtk or FreeSurfer format)")
    parser.add_argument("--data-norm", action="store_true", help="Z-score+prctile data normalization")
    parser.add_argument("--preload", type=str, choices=["none", "cpu", "device"], default="device", help="Data preloading")
    parser.add_argument("--in-ch", type=str, default=["curv", "sulc", "inflated.H"], nargs="+", help="List of geometry")
    parser.add_argument("--hemi", type=str, nargs="+", choices=["lh", "rh"], help="Hemisphere for learning", required=True)
    
    # [수정] data-dir 대신 csv/surf-dir 사용
    parser.add_argument("--csv-file", type=str, default="/data/human/OASIS/OASIS2/oasis_longitudinal_demographics.xlsx", required=True, help="OASIS longitudinal Excel 파일 경로")
    parser.add_argument("--surf-dir", type=str, default="/data/human/OASIS/OASIS2/Freesurfer", required=True, help="Freesurfer 표면 데이터가 있는 기본 디렉토리")
    
    parser.add_argument("--classes", type=int, nargs="+", help="List of regions of interest")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data shuffling")
    parser.add_argument("--aug", type=int, default=0, help="Level of data augmentation")


    parser.add_argument("--test-split-ratio", type=float, default=0.2,help="테스트셋으로 분리할 피험자 비율 (e.g., 0.2 = 20%)")
    
    # (이하 나머지 args는 동일)
    # Training and evaluation
    parser.add_argument("--train-num-steps", type=int, default=20, help="Max epoch")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=8e-5, help="Initial learning rate")
    parser.add_argument("--no-decay", action="store_true", help="Disable decay (every 2 epochs if no progress)")
    parser.add_argument("--loss", type=str, default="dl", choices=["dl", "ce"], help="dl: Dice loss, ce: cross entropy")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Path to the log files (output)")
    parser.add_argument("--ckpt-dir", type=str, default="./logs", help="Path to the checkpoint file (output)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint (pth) to resume training")
    parser.add_argument("--save-and-sample-every", type=int, default=1000)
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
    # Machine settings
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for training (normally, starting with 0)")
    parser.add_argument("--no-cuda", action="store_true", help="No CUDA")
    parser.add_argument("--threads", type=int, default=1, help="# of CPU threads for basis reconstruction")

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

    # 🌟 --- [수정] 통합 ID 맵  --- 🌟
    print(f"사전 로드 (통합 ID 맵 생성용): {args.csv_file}")
    try:
        # 1. 전체 데이터(full_df)를 먼저 로드
        full_df = pd.read_excel(args.csv_file)
        full_df = full_df.dropna(subset=['Subject ID'])
    except Exception as e:
        raise FileNotFoundError(f"Excel 파일 로드 실패: {args.csv_file} - {e}")

    # 2. *전체* 피험자 기준의 통합 ID 맵 생성
    all_subjects_sorted = sorted(full_df['Subject ID'].unique())
    total_unique_subjects = len(all_subjects_sorted)
    
    # 3. 이것이 유일한 ID 맵이 되어야 함
    subject_to_id_map = {subj: i for i, subj in enumerate(all_subjects_sorted)}
    
    print(f"총 {total_unique_subjects}명의 고유 피험자에 대해 통합 ID 맵 생성 완료.")

    map_file_path = 'subject_to_id_map.json'
    print(f"Subject ID 맵을 '{map_file_path}' 파일로 저장합니다...")
    with open(map_file_path, 'w', encoding='utf-8') as f:
        json.dump(subject_to_id_map, f, ensure_ascii=False, indent=4)
    print("저장 완료.")


    # 4. *전체* 피험자 기준으로 훈련/테스트 분리 (문자열 기준)
    rng = np.random.RandomState(args.seed)
    shuffled_subjects = list(all_subjects_sorted) # 복사본 생성
    rng.shuffle(shuffled_subjects)
    
    split_idx = int(total_unique_subjects * (1.0 - args.test_split_ratio))
    train_subjects_set = set(shuffled_subjects[:split_idx])
    test_subjects_set = set(shuffled_subjects[split_idx:])
    
    # 5. *전체* 데이터 기준 max_time_obs 계산
    full_df['MR Delay'] = full_df['MR Delay'].apply(lambda x: str(x).replace('M', '').strip())
    full_df['MR Delay'] = pd.to_numeric(full_df['MR Delay'], errors='coerce').fillna(0)
    max_time_obs = full_df['MR Delay'].max()
    # --- [수정 완료] ---


    # [수정] 훈련 데이터셋(ds_train) 생성
    # Dataset 클래스 호출 순서는 __init__ 정의 순서와 일치해야 합니다.
    ds_train = OASISLongitudinalDataset(
        csv_path=args.csv_file,
        surf_base_dir=args.surf_dir,
        in_ch=args.in_ch,
        hemi=args.hemi,
        # --- 🌟 [수정] 순서 변경 ---
        subject_to_id_map=subject_to_id_map,      # 👈 통합 맵 전달
        subject_partition_set=train_subjects_set, # 👈 훈련용 피험자 목록 전달
        max_time_obs_overall=max_time_obs,        # 👈 전체 최대 시간 전달
        # --- 🌟 기본값 인자 ---
        data_norm=args.data_norm,
        preload=preload,
        data_normalization=args.data_normalization,
        sphere_path=sphere,
        partition='train'
        # -----------------
    )

    # 🌟 [추가] 테스트 데이터셋(ds_test) 생성
    ds_test = OASISLongitudinalDataset(
        csv_path=args.csv_file,
        surf_base_dir=args.surf_dir,
        in_ch=args.in_ch,
        hemi=args.hemi,
        # --- 🌟 [수정] 순서 변경 ---
        subject_to_id_map=subject_to_id_map,      # 👈 동일한 통합 맵 전달
        subject_partition_set=test_subjects_set,  # 👈 테스트용 피험자 목록 전달
        max_time_obs_overall=max_time_obs,        # 👈 전체 최대 시간 전달
        # --- 🌟 기본값 인자 ---
        data_norm=args.data_norm,
        preload="none", 
        data_normalization=args.data_normalization,
        sphere_path=sphere,
        partition='test'
        # -----------------
    )


    # 🌟 [수정] model_in_channels 변수 정의
    model_in_channels = len(args.in_ch) * len(args.hemi)
    if model_in_channels == 0:
        raise ValueError("입력 채널(--in-ch)과 반구(--hemi)를 1개 이상 지정해야 합니다.")

    print(f"모델에 전달할 총 고유 피험자 수: {total_unique_subjects}")
    
    # 🌟 [수정] 수정된 SPHARM_Net 호출
    model = SPHARM_Net(
        sphere=sphere,
        device=device,
        in_ch=model_in_channels,    # (수정)
        n_class=model_in_channels,  # (수정) - 출력도 동일하다고 가정
        C=args.channel,
        L=args.bandwidth,
        D=args.depth,
        interval=args.interval,
        self_condition=False,
        verbose=False,
        add_xyz=True,
        # --- (신규) 종단적 모델을 위한 인자 전달 ---
        max_time_obs=max_time_obs,           # 👈 수정된 변수 사용
        num_subjects=total_unique_subjects   # 👈 수정된 변수 사용
        # -------------------------------------
    )
    model.to(device)

    print("train: auto_normalize: ", args.auto_normalize)
    print("train: data_normalization", args.data_normalization)
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
        dataset=ds_train, # 🌟 ds_train 객체를 전달
        train_batch_size=args.batch_size,
        gradient_accumulate_every=2,
        train_lr=args.learning_rate,
        train_num_steps=args.train_num_steps,
        ema_decay=0.995,
        results_folder=args.results_dir,
        save_and_sample_every=args.save_and_sample_every,
        num_samples=args.num_samples,
        log_dir=args.log_dir,
        valid_dataset=ds_test # 🌟 ds_test 객체를 전달
    )
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    main(args)
