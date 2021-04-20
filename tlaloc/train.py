import json
import torch
from data import StockDataModule
from model import StockGRUModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def main():
    sdm = StockDataModule(data_dir='../data', stock='MSFT')
    model = StockGRUModel(hidden_dim=64, num_layers=4)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='../output/checkpoints',
        filename='tlaloc-{epoch:02d}-{val_loss:.4f}',
        save_top_k=5,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=50,
        verbose=False,
        mode='min'
    )

    trainer = Trainer(gpus=1, 
                      default_root_dir='../output',
                      callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(model, sdm)
    
    # get best checkpoint
    model = StockGRUModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    with open('../output/models/transforms.json', 'w') as f:
        f.write(json.dumps(sdm.scaler, indent=4))

    torch.save(model.state_dict(), '../output/models/model.pth')
    
if __name__ == '__main__':
    main()
