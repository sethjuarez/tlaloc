import os
import sys
import json
import math
import torch
import shutil
import mlflow
import warnings
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from model import EarningsGRUModel
from pytorch_lightning import Trainer
from typing import List, Dict, Optional, Any
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

class MLFlowAutoLogger(MLFlowLogger):
    def __init__(self, **kwargs):
        try:
            from azureml.core import Run
            self.run = Run.get_context()
            ws = run.experiment.workspace
            super().__init__(experiment_name=self.run.experiment.name, 
                                tracking_uri=ws.get_mlflow_tracking_uri())
            print('Using AzureML MLFlow Tracking')
        except Exception as e:
            super().__init__(**kwargs)
            self.run = None
            print('Using Standard MLFlow Tracking')

    def log_image(self, name, path=None, plot=None, description=''):
        if self.run != None:
            self.run.log_image(name, path, plot, description)


class EarningsCLI(LightningCLI):

    def run_simulation(self, sequence: torch.Tensor, model: EarningsGRUModel, 
                        window: int, emin: float, emax: float, output_dir: Path):

        print('Running Simulation...')
        # caluate lookback and forward
        behind = math.floor(len(sequence) / 2)
        forward = math.floor(behind / 2)

        # predict
        p_seq = model.predict(list(sequence.numpy()), window, behind, forward)

        # inverse scale
        p_seq = SeqDataset.inverse_scale(torch.FloatTensor(p_seq), emin, emax)
        r_seq = SeqDataset.inverse_scale(torch.FloatTensor(sequence), emin, emax)

        # create indices
        seq_size = sequence.size(0)
        actual_range = [i for i in range(seq_size)]
        pred_range = [i for i in range(seq_size-behind,seq_size+forward)]

        # plot predictions over actuals
        fig, ax = plt.subplots(figsize=(1600/96., 600/96.))
        ax.plot(actual_range, r_seq, color='blue', label='actual')
        ax.plot(pred_range, p_seq, color='red', label='predictions')
        ax.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, p: '${:1.0f}K'.format(x/1000.)))
        leg = ax.legend()
        plt.title(f'Sequence Prediction [Back: {behind}, Forward: {forward}]')
        plt.xlabel('Period')
        plt.ylabel('Earnings')
        
        img_file = str(output_dir / 'validation_inf_run.png')
        print(f'Saving simulation to {img_file}')

        # save image
        plt.savefig(img_file, dpi=96)

        return str(img_file)
        
    def save_model(self, base_dirs: List[Path], model: EarningsGRUModel, model_params: Dict):
        for dirs in base_dirs:
            with open(str(dirs / 'params.json'), 'w') as f:
                f.write(json.dumps(model_params, indent=4))
            torch.save(model.state_dict(), str(dirs / 'model.pth'))

    def instantiate_trainer(self) -> None:
        # start mlflow autologging
        mlflow.autolog()

        # get trainer going
        super().instantiate_trainer()

        # get default root log directory
        default_root_dir = check_dir(Path(self.trainer.default_root_dir).resolve())
        print(default_root_dir)

        # adding additional loggers
        tb_logger = TensorBoardLogger(save_dir=str(default_root_dir), name='logs')
        self.mlf_logger = MLFlowLogger(experiment_name="earnings-experiment", 
                                        tracking_uri=f"file:{str(default_root_dir / 'mlflow')}")
        
        self.trainer.logger = LoggerCollection([tb_logger, self.mlf_logger])

        # set versioned directory
        self.tb_version = tb_logger.version
        self.tb_log_dir = Path(tb_logger.log_dir).resolve()

        # resetting output directories for callbacks
        for cb in self.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.dirpath = f'{Path(tb_logger.log_dir).resolve() / "checkpoints"}'
            if isinstance(cb, SaveConfigCallback):
                cb.config_filename = f'{Path(tb_logger.log_dir).resolve() / "config.yaml"}'

    def after_fit(self):
        # output paths
        output_dir = Path(self.trainer.default_root_dir).resolve()
        
        # get best model from checkpoints
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        model = EarningsGRUModel.load_from_checkpoint(best_model_path)
        
        # additional model params for inference
        model_params = { 
            'model': model.hparams, 
            'data': self.trainer.datamodule.metadata
        }

        # save model and inference paramters (latest and versioned)
        model_dir = check_dir(output_dir / 'model')
        model_version_dir = check_dir(model_dir / f'version_{self.tb_version}')
        self.save_model([model_version_dir, model_dir], model, model_params)

        default_root_dir = check_dir(Path(self.trainer.default_root_dir).resolve())

        # plot chart with validation data for run review
        datap = model_params['data']
        val_seq = self.trainer.datamodule.val_dataset.seq
        image_file = self.run_simulation(val_seq, model, datap['window'], 
                                            datap['min'], datap['max'], 
                                            default_root_dir)
        

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    EarningsCLI(EarningsGRUModel, EarningsDataModule)