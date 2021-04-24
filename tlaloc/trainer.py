from model import EarningsGRUModel
from data import SeqDataset, EarningsDataModule
from pytorch_lightning.utilities.cli import LightningCLI


LightningCLI(EarningsGRUModel, EarningsDataModule)