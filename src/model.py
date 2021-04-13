import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F

class StockGRUModel(pl.LightningModule):
    def __init__(self, input_dim: int = 1, 
                       hidden_dim: int = 32,
                       num_layers: int = 2, 
                       output_dim: int = 1, 
                       lr: float = 0.5):
        super().__init__()

        self.to
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.gru(x, (h0.detach().to(self.device)))
        out = self.fc(out[:, -1, :])
        return out

    def _step(self, x: torch.Tensor, y: torch.Tensor):
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        acc = ((y_hat - y) / y).mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(*batch)
        self.log('dev', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(*batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dev', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

if __name__ == '__main__':
    pass