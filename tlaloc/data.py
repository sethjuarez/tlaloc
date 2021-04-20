import math
import torch
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
        return self.seq[i:e].view((self.win, 1)), self.seq[e:e+1]

    @staticmethod
    def scale(sequence: torch.Tensor, min: float, max: float) -> torch.Tensor:
        return (sequence-min)/(max-min)

    @staticmethod
    def inverse_scale(sequence: torch.Tensor, min: float, max: float) -> torch.Tensor:
        return (sequence*(max - min)) + min

class StockDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'data',
                       stock: str = 'MSFT',
                       lookback: int = 100,
                       batch_size: int = 512,
                       train_split: float = .9,
                       start: datetime = None, 
                       end: datetime = None):
        super().__init__()

        # storage
        self.data_dir = data_dir

        # stock data
        self.stock = stock
        self.start = start
        self.end = end

        # training parameters
        self.lookback = lookback
        self.batch_size = batch_size
        self.test_split = 1 - train_split

    @staticmethod
    def get_stock_data(data_dir: str = 'data',
                       stock: str = 'MSFT',
                       start: datetime = None, 
                       end: datetime = None):

        base = 'https://query1.finance.yahoo.com/v7/finance/download'
        post = 'interval=1d&events=history&includeAdjustedClose=true'

        data_dir = Path(data_dir).resolve()

        beg = datetime(1970, 1, 1)
        s = beg if start == None else start
        e = datetime.now() if end == None else end

        filename = f'{stock}_{s.strftime("%y.%m.%d")}_{e.strftime("%y.%m.%d")}.parquet'
        filepath = Path(data_dir / filename)

        df: pd.DataFrame = None
        if not filepath.exists():
            sdate = int((s - beg).total_seconds())
            edate = int((e - beg).total_seconds())
            url = f'{base}/{stock}?period1={sdate}&period2={edate}&{post}'
            df = pd.read_csv(url, parse_dates=True)
            return df.to_parquet(str(filepath))
        else:
            return pd.read_parquet(str(filepath))

    def prepare_data(self):
        # will download if it's not there
        StockDataModule.get_stock_data(self.data_dir, self.stock, self.start, self.end)

    def setup(self, stage: Optional[str] = None):
        # get latest parque file written
        df = StockDataModule.get_stock_data(self.data_dir, self.stock, self.start, self.end)
        data_all = df['Close'].values

        test_sz = math.floor(self.test_split * len(data_all))

        train_data = torch.FloatTensor(data_all[:-test_sz]).view(-1)
        val_data = torch.FloatTensor(data_all[-test_sz:]).view(-1)

        # Min/Max Scaling from Training Data
        cmin, cmax = train_data.min(), train_data.max()
        self.scaler = { "min": float(cmin), "max": float(cmax) }

        # create sequence datasets (scaled)
        self.train_dataset = SeqDataset(SeqDataset.scale(train_data, cmin, cmax), 
                                                self.lookback)

        self.val_dataset = SeqDataset(SeqDataset.scale(val_data, cmin, cmax), 
                                                self.lookback)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

if __name__ == '__main__':
    import numpy as np
    seq = torch.from_numpy(np.array([float(i) for i in range(5)]))
    print(seq)
    sdataset = SeqDataset(seq, 3)
    for i in range(len(sdataset)):
        print(sdataset[i])

    sdm = StockDataModule(data_dir='../data', stock='MSFT')
    sdm.prepare_data()
    sdm.setup()
    print(sdm.transform)
