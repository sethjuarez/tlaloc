import os
import json
import torch
import shutil
import warnings
from pathlib import Path
from model import EarningsGRUModel
from pytorch_lightning import Trainer
from typing import List, Dict
from data import SeqDataset, EarningsDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.loggers import LoggerCollection, MLFlowLogger, TensorBoardLogger

def check_dir(dir: Path, clear=False):
    if dir.exists() and clear:
        shutil.rmtree(str(dir))
    if not dir.exists():
        os.makedirs(str(dir))
    return dir

class EarningsCLI(LightningCLI):
    
    def save_model(self, base_dirs: List[Path], model: EarningsGRUModel, model_params: Dict):
        for dirs in base_dirs:
            with open(str(dirs / 'params.json'), 'w') as f:
                f.write(json.dumps(model_params, indent=4))
            torch.save(model.state_dict(), str(dirs / 'model.pth')) 

    def instantiate_trainer(self) -> None:
        # get trainer going
        super().instantiate_trainer()

        # get default root log directory
        default_root_dir = self.trainer.default_root_dir

        
        # adding additional loggers
        tb_logger = TensorBoardLogger(save_dir=default_root_dir, name='logs')
        mlf_logger = MLFlowLogger(experiment_name="default", 
                        tracking_uri=f"file:{default_root_dir}/mlflow")

        self.trainer.logger = LoggerCollection([tb_logger, mlf_logger])

        # set versioned directory
        self.tb_version = tb_logger.version

        # resetting output directories for callbacks
        for cb in self.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.dirpath = f'{tb_logger.log_dir}/checkpoints'
            if isinstance(cb, SaveConfigCallback):
                cb.config_filename = f'{tb_logger.log_dir}/config.yaml'

    def after_fit(self):
        trainer = self.trainer
        # output paths
        output_dir = Path(trainer.default_root_dir).resolve()
        
        # get best model from checkpoints
        best_model_path = trainer.checkpoint_callback.best_model_path
        model = EarningsGRUModel.load_from_checkpoint(best_model_path)
        
        # additional model params for inference
        model_params = { 
            'model': model.hparams, 
            'data': trainer.datamodule.metadata
        }

        # save model and inference paramters (latest and versioned)
        model_dir = check_dir(output_dir / 'model')
        model_version_dir = check_dir(model_dir / f'version_{self.tb_version}')
        self.save_model([model_version_dir, model_dir], model, model_params)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    EarningsCLI(EarningsGRUModel, EarningsDataModule)