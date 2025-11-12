from __future__ import annotations
import pandas as pd
import numpy as np
import torch
from typing import Optional, Dict

from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.torch_datasets import MyTensorDataset
from src.utilities.utils import get_logger

log = get_logger(__name__)

def create_sliding_window_samples(
    data: np.ndarray, 
    window: int, 
    horizon: int, 
    multi_horizon: bool = False
) -> Dict[str, np.ndarray]:
    """
    1D 시계열 데이터로부터 (입력, 타겟) 슬라이딩 윈도우 샘플을 생성합니다.
    """
    seq_len = len(data)
    inputs = []
    targets = []

    # 전체 시퀀스를 순회하며 (입력 윈도우, 타겟) 쌍을 만듭니다.
    for i in range(seq_len - window - horizon + 1):
        input_window = data[i : i + window]
        
        if multi_horizon:
            # 여러 스텝을 예측해야 하는 경우 (예: [t+1, t+2, ... t+horizon])
            target_window = data[i + window : i + window + horizon]
            targets.append(target_window)
        else:
            # 단일 스텝(t+horizon)만 예측하는 경우
            target_step = data[i + window + horizon - 1]
            targets.append(target_step)
            
        inputs.append(input_window)

    if not inputs:
        log.warning("데이터 기간이 너무 짧아서 샘플을 생성할 수 없습니다.")
        # 빈 배열을 올바른 차원으로 반환
        if multi_horizon:
            return {"inputs": np.empty((0, window, 1)), "targets": np.empty((0, horizon, 1))}
        else:
            return {"inputs": np.empty((0, window, 1)), "targets": np.empty((0, 1))}

    # Dyffusion 모델은 [Batch, Time, Channel] 3D 입력을 기대합니다.
    # [num_samples, window] -> [num_samples, window, 1]
    inputs_np = np.array(inputs)[:, :, np.newaxis] 
    
    if multi_horizon:
        # [num_samples, horizon] -> [num_samples, horizon, 1]
        targets_np = np.array(targets)[:, :, np.newaxis]
    else:
        # [num_samples] -> [num_samples, 1] (채널 차원 불필요)
        targets_np = np.array(targets)

    return {"inputs": inputs_np, "targets": targets_np}


class StockPriceDataModule(BaseDataModule):
    """
    CSX.csv 같은 1D 주식 가격 시계열을 위한 데이터 모듈
    """
    def __init__(
        self,
        # --- 수정된 부분: 파일 경로를 직접 받음 ---
        train_file_path: str,
        val_file_path: str,
        test_file_path: str,
        # ------------------------------------
        
        target_column: str = "Close",
        csv_header_row: int = 2, # CSX.csv 형식(헤더가 3번째 줄)에 맞춘 기본값
        
        # 모델 입력/예측 설정
        window: int = 16,     # 입력으로 사용할 과거 일수
        horizon: int = 1,       # 예측할 미래 시점 (1일 뒤)
        multi_horizon: bool = False, # 여러 스텝을 예측할지 여부
        
        # 정규화 (Normalization)
        normalize: bool = True,
        
        **kwargs,
    ):
        # BaseDataModule 초기화 (batch_size, num_workers 등 전달)
        super().__init__(data_dir=".", **kwargs) 
        # file_path 등 이 클래스 고유의 하이퍼파라미터 저장
        self.save_hyperparameters(
            "train_file_path", "val_file_path", "test_file_path",
            "target_column", "csv_header_row",
            "window", "horizon", "multi_horizon", "normalize"
        )
        
        self.scaler = None # 정규화를 위한 Scaler

    def _load_and_process_series(self, file_path: str) -> pd.Series:
        """지정된 CSV 파일을 로드하고 타겟 열을 Series로 반환합니다."""
        try:
            df = pd.read_csv(file_path, header=self.hparams.csv_header_row)
        except FileNotFoundError:
            log.error(f"파일을 찾을 수 없습니다: {file_path}")
            raise
        
        # 날짜(Date) 열 처리
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        
        # 예측할 타겟 열 선택 (예: 'Close' 종가)
        return df[self.hparams.target_column].dropna()

    def setup(self, stage: Optional[str] = None):
        """
        데이터 로드 및 분할 (train/val/test)
        """
        
        # 1. 정규화(Normalization) 기준을 위해 *항상* 학습(train) 파일부터 로드
        log.info(f"Loading train data for normalization stats from: {self.hparams.train_file_path}")
        train_series = self._load_and_process_series(self.hparams.train_file_path)
        
        if self.hparams.normalize:
            # (X - min) / (max - min) 정규화
            self.data_min = train_series.min()
            self.data_max = train_series.max()
            self.data_range = self.data_max - self.data_min
            
            # 정규화 함수 정의
            def normalize_func(series):
                return (series - self.data_min) / self.data_range
            
            log.info(f"데이터 정규화 기준 설정 완료 (Min: {self.data_min}, Max: {self.data_max})")
        else:
            # 정규화 안 함 (데이터 그대로 사용)
            def normalize_func(series):
                return series

        # 2. 스테이지에 맞게 데이터 로드 및 처리
        if stage == "fit" or stage is None:
            # 학습 데이터
            train_norm_series = normalize_func(train_series)
            train_samples = create_sliding_window_samples(
                train_norm_series.values, self.hparams.window, self.hparams.horizon, self.hparams.multi_horizon
            )
            self._data_train = MyTensorDataset(
                {k: torch.from_numpy(v).float() for k, v in train_samples.items()}, dataset_id='train'
            )
            log.info(f"Train data loaded: {len(train_norm_series)} points, {len(self._data_train)} samples.")
            
            # 검증 데이터
            log.info(f"Loading validation data from: {self.hparams.val_file_path}")
            val_series = self._load_and_process_series(self.hparams.val_file_path)
            val_norm_series = normalize_func(val_series)
            val_samples = create_sliding_window_samples(
                val_norm_series.values, self.hparams.window, self.hparams.horizon, self.hparams.multi_horizon
            )
            self._data_val = MyTensorDataset(
                {k: torch.from_numpy(v).float() for k, v in val_samples.items()}, dataset_id='val'
            )
            log.info(f"Validation data loaded: {len(val_norm_series)} points, {len(self._data_val)} samples.")

        if stage == "test" or stage is None:
            # 테스트 데이터
            log.info(f"Loading test data from: {self.hparams.test_file_path}")
            test_series = self._load_and_process_series(self.hparams.test_file_path)
            test_norm_series = normalize_func(test_series)
            test_samples = create_sliding_window_samples(
                test_norm_series.values, self.hparams.window, self.hparams.horizon, self.hparams.multi_horizon
            )
            self._data_test = MyTensorDataset(
                {k: torch.from_numpy(v).float() for k, v in test_samples.items()}, dataset_id='test'
            )
            log.info(f"Test data loaded: {len(test_norm_series)} points, {len(self._data_test)} samples.")

        self.print_data_sizes(stage)
