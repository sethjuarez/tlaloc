import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# consider multiple stocks
# to perform deep learned 
# statistical arbitrage
class SeqDataset(Dataset):
    def __init__(self, sequence: torch.Tensor, window: int) -> None:
        self.sz = len(sequence) - window
        self.win = window
        self.seq = sequence

    def __len__(self) -> int:
        return self.sz

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        e = i + self.win
        return self.seq[i:e], self.seq[e:e+1]

    @staticmethod
    def scale(sequence: torch.Tensor, min: float, max: float) -> torch.Tensor:
        return (sequence - min) / (max - min)

    @staticmethod
    def inverse_scale(sequence: torch.Tensor, min: float, max: float) -> torch.Tensor:
        return (sequence*(max - min)) + min


class EarningsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'path/to/data', parquet: str = 'parquet_file', 
                    resource_id: int = -1, window: int = 100, batch_size: int = 64, 
                    train_split: float = .80):
        super().__init__()

        # path to dataframe
        self.data_dir = Path(data_dir).resolve()
        self.parquet = self.data_dir / parquet
        self.resource_id = resource_id

        # training parameters
        self.window = window
        self.batch_size = batch_size
        self.test_split = 1 - train_split

    @staticmethod
    def load(parquet_file: str = 'path/to/parquet', resource_id: int = -1) -> np.array:
        # get data
        df = pd.read_parquet(parquet_file)

        # get appropriate data (all aggregate or id)
        rid, datestr, earnings = 'resource_id', 'date', 'earnings'
        res_min, res_max = df[rid].min(), df[rid].max()
        if resource_id >= res_min and resource_id <= res_max:
            df = df.loc[df[rid] == 1]
        else:
            df = df.groupby(by=[datestr]).sum()

        df = df.sort_values(by=[datestr])
        return df[earnings].values


    def setup(self, stage: Optional[str] = None):
        # get data
        data_all = EarningsDataModule.load(self.parquet)

        # min/max for scaling
        cmin, cmax = data_all.min(), data_all.max()
        self.metadata = { 
            "min": float(cmin), 
            "max": float(cmax), 
            "window": self.window 
        }

        # scale
        data_all = SeqDataset.scale(data_all, cmin, cmax)

        # data split
        test_sz = math.floor(self.test_split * len(data_all))
        train_data = torch.FloatTensor(data_all[:-test_sz])
        val_data = torch.FloatTensor(data_all[-test_sz:])

        # create sequence datasets
        self.train_dataset = SeqDataset(train_data, self.window)
        self.val_dataset = SeqDataset(val_data, self.window)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

if __name__ == '__main__':
    sdm = EarningsDataModule(data_dir='../data', parquet='sales.parquet')
    sdm.setup()
    print(sdm.metadata)
