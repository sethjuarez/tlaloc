from model import StockGRUModel
from data import SeqDataset, StockDataModule
from pytorch_lightning.utilities.cli import LightningCLI


LightningCLI(StockGRUModel, StockDataModule)