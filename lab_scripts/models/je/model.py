import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import numpy as np



class JEAutoencoder(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
    
    def forward(self, x):
        pass

    def training_step(self, batch, batch_n):
        pass