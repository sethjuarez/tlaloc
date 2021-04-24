import os
import json
import torch
import shutil
from pathlib import Path
from model import EarningsGRUModel
from data import SeqDataset, EarningsDataModule
from pytorch_lightning.utilities.cli import LightningCLI

def check_dir(dir: Path, clear=False):
    if dir.exists() and clear:
        shutil.rmtree(str(dir))
    if not dir.exists():
        os.makedirs(str(dir))
    return dir

class EarningsTrainer(LightningCLI):
    def before_fit(self):
        root_dir = check_dir(Path(self.trainer.default_root_dir).resolve())
        print(f'default_root_dir: {root_dir}')

    def after_fit(self):
        trainer = self.trainer
        # output paths
        output_dir = Path(trainer.default_root_dir).resolve()
        model_dir = check_dir(output_dir / 'model')

        # get best model from checkpoints
        best_model_path = trainer.checkpoint_callback.best_model_path
        model = EarningsGRUModel.load_from_checkpoint(best_model_path)
        
        # additional model params for inference
        model_params = { 
            'model': model.hparams, 
            'data': trainer.datamodule.metadata
        }

        # save model and params
        with open(str(model_dir / 'params.json'), 'w') as f:
            f.write(json.dumps(model_params, indent=4))

        torch.save(model.state_dict(), str(model_dir / 'model.pth'))        

if __name__ == '__main__':
    EarningsTrainer(EarningsGRUModel, EarningsDataModule)