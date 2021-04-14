import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from datetime import datetime

class Ticker:
    def __init__(self, data_dir: str = 'data',
                       stock: str = 'MSFT',
                       start: datetime = None, 
                       end: datetime = None):

        self._beg = datetime(1970, 1, 1)
        self.data_dir = Path(data_dir).resolve()

        self.start = self._beg if start == None else start
        self.end = datetime.now() if end == None else end

        self.base = 'https://query1.finance.yahoo.com/v7/finance/download'
        self.post = 'interval=1d&events=history&includeAdjustedClose=true'

        fmt = "%y.%m.%d"
        self.filename = f'{stock}_{self.start.strftime(fmt)}_{self.end.strftime(fmt)}.parquet'
        self.filepath = Path(self.data_dir / self.filename)

        df: pd.DataFrame = None
        if not self.filepath.exists():
            sdate = int((self.start - self._beg).total_seconds())
            edate = int((self.end - self._beg).total_seconds())
            url = f'{self.base}/{stock}?period1={sdate}&period2={edate}&{self.post}'
            df = pd.read_csv(url, parse_dates=True)
            df.to_parquet(str(self.filepath))
        else:
            df = pd.read_parquet(str(self.filepath))

        self.dataframe = df

class TickerDataset(Dataset):
    def __init__():
        pass

    def __getitem__(self, index) -> torch.Tensor:
        return super().__getitem__(index)

    def __len__() -> int:
        return 1

if __name__ == '__main__':
    tick = Ticker('../data', 'MSFT')
    print(tick.filename)
    print(tick.dataframe)
