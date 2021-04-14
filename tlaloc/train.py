from datetime import datetime

import torch
import json
from ticker import Ticker
from data import StockDataModule
from model import StockGRUModel
from pytorch_lightning import Trainer

def main():
    tick = Ticker('../data', 'MSFT')
    sdm = StockDataModule(tick)
    model = StockGRUModel()
    trainer = Trainer(gpus=1, max_epochs=3, default_root_dir='../logs')
    trainer.fit(model, sdm)

    with open('../models/transforms.json', 'w') as f:
        f.write(json.dumps(sdm.transform, indent=4))

    model.to_onnx('../models/model.onnx', export_params=True)
    
if __name__ == '__main__':
    main()
