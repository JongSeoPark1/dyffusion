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
        file_path: str,
        target_column: str = "Close",
        
        # 10년 단위 분할 설정
        train_start_year: int = 1981,
        train_end_year: int = 1989,
        val_start_year: int = 1990,
        val_end_year: int = 1999,
        test_start_year: int = 2000,
        test_end_year: int = 2009,
        
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
            "file_path", "target_column",
            "train_start_year", "train_end_year",
            "val_start_year", "val_end_year",
            "test_start_year", "test_end_year",
            "window", "horizon", "multi_horizon", "normalize"
        )
        
        self.scaler = None # 정규화를 위한 Scaler

    def setup(self, stage: Optional[str] = None):
        """
        데이터 로드 및 분할 (train/val/test)
        """
        try:
            # 1. CSV 파일 로드 (CSX.csv 형식에 맞게 header=2)
            df = pd.read_csv(self.hparams.file_path, header=2)
        except FileNotFoundError:
            log.error(f"파일을 찾을 수 없습니다: {self.hparams.file_path}")
            raise
            
        # 2. 날짜(Date) 열 처리
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        
        # 3. 예측할 타겟 열 선택 (예: 'Close' 종가)
        series = df[self.hparams.target_column].dropna()

        # 4. 정규화 (Normalization)
        # 중요: 정규화는 *반드시* 학습 데이터(train) 기준으로 수행해야 함
        train_data = series.loc[
            f"{self.hparams.train_start_year}":f"{self.hparams.train_end_year}"
        ]
        
        if self.hparams.normalize:
            # (X - min) / (max - min) 정규화
            self.data_min = train_data.min()
            self.data_max = train_data.max()
            self.data_range = self.data_max - self.data_min
            
            series = (series - self.data_min) / self.data_range
            log.info(f"데이터 정규화 완료 (Min: {self.data_min}, Max: {self.data_max})")

        # 5. 10년 단위로 데이터 분할 (남는 데이터는 버려짐)
        if stage == "fit" or stage is None:
            train_series = series.loc[
                f"{self.hparams.train_start_year}":f"{self.hparams.train_end_year}"
            ]
            val_series = series.loc[
                f"{self.hparams.val_start_year}":f"{self.hparams.val_end_year}"
            ]
            
            log.info(f"Train data: {len(train_series)} points ({self.hparams.train_start_year}-{self.hparams.train_end_year})")
            log.info(f"Val data:   {len(val_series)} points ({self.hparams.val_start_year}-{self.hparams.val_end_year})")

            # 6. 슬라이딩 윈도우 샘플 생성
            train_samples = create_sliding_window_samples(
                train_series.values, self.hparams.window, self.hparams.horizon, self.hparams.multi_horizon
            )
            val_samples = create_sliding_window_samples(
                val_series.values, self.hparams.window, self.hparams.horizon, self.hparams.multi_horizon
            )

            # 7. Tensor Dataset으로 변환
            self._data_train = MyTensorDataset(
                {k: torch.from_numpy(v).float() for k, v in train_samples.items()}, dataset_id='train'
            )
            self._data_val = MyTensorDataset(
                {k: torch.from_numpy(v).float() for k, v in val_samples.items()}, dataset_id='val'
            )

        if stage == "test" or stage is None:
            test_series = series.loc[
                f"{self.hparams.test_start_year}":f"{self.hparams.test_end_year}"
            ]
            log.info(f"Test data:  {len(test_series)} points ({self.hparams.test_start_year}-{self.hparams.test_end_year})")
            
            test_samples = create_sliding_window_samples(
                test_series.values, self.hparams.window, self.hparams.horizon, self.hparams.multi_horizon
            )
            self._data_test = MyTensorDataset(
                {k: torch.from_numpy(v).float() for k, v in test_samples.items()}, dataset_id='test'
            )

        # (predict 스테이지는 생략)

        self.print_data_sizes(stage)
