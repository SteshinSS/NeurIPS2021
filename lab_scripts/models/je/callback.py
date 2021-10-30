import anndata as ad
import pytorch_lightning as pl
import torch
from lab_scripts.metrics import je as metrics


class TargetCallback(pl.Callback):
    def __init__(self, solution: ad.AnnData, dataset, frequency: int):
        self.solution = solution
        self.dataset = dataset
        self.frequency = frequency
        self.current_epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        logger = trainer.logger
        if not logger:
            return
        self.current_epoch += 1
        if self.current_epoch % self.frequency != 0:
            return
        pl_module.eval()
        embeddings = []
        device = pl_module.device
        with torch.no_grad():
            for i, batch in enumerate(self.dataset):
                first, second = batch
                embeddings.append(pl_module(first.to(device), second.to(device)).cpu())
        embeddings = torch.cat(embeddings, dim=0)
        prediction = metrics.create_anndata(self.solution, embeddings.numpy())
        all_metrics = metrics.calculate_metrics(prediction, self.solution)
        logger.experiment.log(all_metrics)
