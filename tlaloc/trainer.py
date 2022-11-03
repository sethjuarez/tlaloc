import os
import sys
import json
import math
import torch
import shutil
import mlflow
import warnings
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from model import EarningsGRUModel
from typing import List, Dict, Optional, Any
from data import SeqDataset, EarningsDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI

def check_dir(dir: Path, clear=False):
    if dir.exists() and clear:
        shutil.rmtree(str(dir))
    if not dir.exists():
        os.makedirs(str(dir))
    return dir

class EarningsCLI(LightningCLI):

    def plot_simulation(self, sequence: torch.Tensor, model: EarningsGRUModel, 
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
        pred_range = [i for i in range(seq_size-behind-1,seq_size+forward-1)]

        # plot predictions over actuals
        fig, ax = plt.subplots(figsize=(1600/96., 600/96.))
        ax.plot(actual_range, r_seq, color='blue', label='actual')
        ax.plot(pred_range, p_seq, color='red', label='predictions')
        ax.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, p: '${:1.0f}K'.format(x/1000.)))
        leg = ax.legend()
        plt.title(f'Sequence Prediction [Back: {behind}, Forward: {forward}]')
        plt.xlabel('Period')
        plt.ylabel('Earnings')
        
        img_file = str(output_dir / 'validation_data.png')
        print(f'Saving simulation to {img_file}')

        # save image
        plt.savefig(img_file, dpi=96)

        return str(img_file)

    def plot_original(self, datamodule: EarningsDataModule(), output_dir: Path):
        df = pd.read_parquet(datamodule.parquet)
        fig, ax = plt.subplots(figsize=(1600/96., 600/96.))
        for key, grp in df.groupby(['resource_id']):
            ax = grp.plot(ax=ax, kind='line', x='date', y='earnings', label=key)

        ax.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, p: '${:1.0f}K'.format(x/1000.)))
        leg = ax.legend()
        plt.title(f'Original Earnings Data')
        plt.xlabel('Period')
        plt.ylabel('Earnings')
        
        img_file = str(output_dir / 'original_data.png')
        print(f'Saving simulation to {img_file}')

        # save image
        plt.savefig(img_file, dpi=96)

        return str(img_file)
        
    def save_model(self, base_dirs: List[Path], model: EarningsGRUModel, model_params: Dict):
        input_sample = torch.zeros((1, 100, 1))
        for dirs in base_dirs:
            with open(str(dirs / 'params.json'), 'w') as f:
                f.write(json.dumps(model_params, indent=4))
            torch.save(model.state_dict(), str(dirs / 'model.pth'))
            model.to_onnx(dirs / 'model.onnx', input_sample, export_params=True)


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
        self.save_model([model_dir], model, model_params)
        default_root_dir = check_dir(Path(self.trainer.default_root_dir).resolve())

        # plot original data
        self.plot_original(self.trainer.datamodule, default_root_dir)
        # plot chart with validation data for run review
        datap = model_params['data']
        val_seq = self.trainer.datamodule.val_dataset.seq
        image_file = self.plot_simulation(val_seq, model, datap['window'], 
                                            datap['min'], datap['max'], 
                                            default_root_dir)
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
        EarningsCLI(EarningsGRUModel, EarningsDataModule)