import os
import json
import torch
import shutil
from pathlib import Path
from datetime import datetime
from model import StockGRUModel
from data import StockDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def check_dir(dir: Path, clear=False):
    if dir.exists() and clear:
        shutil.rmtree(str(dir))
    if not dir.exists():
        os.makedirs(str(dir))

def main(stock='MSFT', data_dir='../data', output_dir='../output'):    
    data_dir = Path(data_dir).resolve()
    check_dir(data_dir)

    output_dir = Path(output_dir).resolve()
    check_dir(output_dir, clear=True)

    now = datetime.now()

    sdm = StockDataModule(data_dir=str(data_dir), stock=stock, 
                            start=datetime(now.year - 5, now.month, now.day))

    model = StockGRUModel(lr=0.01)

    chkpt_dir = output_dir / 'checkpoints'
    check_dir(chkpt_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=str(chkpt_dir),
        filename='tlaloc-{epoch:02d}-{val_dev:.4f}',
        save_top_k=5,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=25,
        verbose=False,
        mode='min'
    )

    trainer = Trainer(gpus=1, default_root_dir=str(output_dir),
                      callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(model, sdm)
    
    # get best checkpoint
    model = StockGRUModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    model_dir = output_dir / 'models'
    check_dir(model_dir)

    model_params = { 
        'stock': stock, 
        'model': model.hparams, 
        'data': sdm.metadata
    }

    with open(str(model_dir / 'params.json'), 'w') as f:
        f.write(json.dumps(model_params, indent=4))

    torch.save(model.state_dict(), str(model_dir / 'model.pth'))
    
if __name__ == '__main__':
    main(stock='MSFT')
