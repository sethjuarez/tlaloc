import torch
import numpy as np
from ticker import Ticker
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

class StockDataModule(pl.LightningDataModule):
    def __init__(self, ticker: Ticker,
                       lookback: int = 100,
                       batch_size: int = 512,
                       train_split: float = .7):
        super().__init__()
        self.ticker = ticker
        self.lookback = lookback
        self.batch_size = batch_size
        self.train_split = train_split

    def setup(self, stage: Optional[str] = None):
        df = self.ticker.dataframe
        cmin = df['Close'].min()
        cmax = df['Close'].max()
        self.transform = {
            "Close": {
                "min": cmin,
                "max": cmax
            }
        }
        
        # scale [0, 1]
        df = df[['Close']]
        data_raw = (df.values - cmin) / (cmax - cmin)
        
        # create all possible sequences of length seq_len
        data = []
        for index in range(len(data_raw) - self.lookback): 
            data.append(data_raw[index: index + self.lookback])

        self.raw_data = np.array(data)
        self.train_set_size = int(np.round(self.train_split*self.raw_data.shape[0]))

        x_train = self.raw_data[:self.train_set_size,:-1,:]
        y_train = self.raw_data[:self.train_set_size,-1,:]
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        self.train_dataset = TensorDataset(x_train, y_train)

        x_test = self.raw_data[self.train_set_size:,:-1]
        y_test = self.raw_data[self.train_set_size:,-1,:]
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)
        self.val_dataset = TensorDataset(x_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

if __name__ == '__main__':
    from datetime import datetime
    tick = Ticker('../data', 'MSFT', datetime(2016, 4, 9), datetime(2021, 4, 9))
    sdm = StockDataModule(tick)
    sdm.setup()
    print(sdm.transform)
