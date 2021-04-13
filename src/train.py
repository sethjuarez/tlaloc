from datetime import datetime
from ticker import Ticker
from data import StockDataModule
from model import StockGRUModel
from pytorch_lightning import Trainer

def main():
    tick = Ticker('../data', 'MSFT')
    sdm = StockDataModule(tick)
    model = StockGRUModel()
    trainer = Trainer(gpus=1)
    trainer.fit(model, sdm)
    
if __name__ == '__main__':
    main()
