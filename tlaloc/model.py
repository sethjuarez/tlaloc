import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import List, Tuple

class EarningsGRUModel(pl.LightningModule):
    def __init__(self, input_dim: int = 1, 
                       hidden_dim: int = 32,
                       num_layers: int = 2, 
                       output_dim: int = 1, 
                       lr: float = 0.5):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        h0 = h0.detach().to(self.device)
        out, _ = self.gru(x.view(-1, x.size(1), 1), (h0))
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, seq: List[float], window: int, lookbehind: int, lookahead: int) -> List[float]:
        overlap = []
        predixn = seq[-window:]
        self.eval()
        with torch.no_grad():
            # get lookbehind period (overlap)
            for i in reversed(range(1, lookbehind+1)):
                x = torch.FloatTensor(seq[-window-i:-i]).view(1, -1)
                y = self(x)
                overlap.append(y.detach().item())

            # begin lookahead period (predictions)
            for i in range(lookahead):
                x = torch.FloatTensor(predixn[-window:]).view(1, -1)
                y = self(x)
                predixn.append(y.detach().item())

        return overlap + predixn[-lookahead:]

    def _step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        dev = ((y_hat - y) / y).mean()
        return loss, dev

    def training_step(self, batch, batch_idx):
        loss, dev = self._step(*batch)
        self.log('dev', dev, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dev = self._step(*batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dev', dev, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_schedule = optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                step_size=10, 
                                                gamma=0.1)
        return [optimizer], [lr_schedule]

if __name__ == '__main__':
    pass