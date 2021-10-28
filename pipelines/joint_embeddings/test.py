import anndata as ad
from lab_scripts.metrics.je import metrics
import logging
logging.basicConfig(level=logging.INFO)

predictions = ad.read_h5ad('output.h5ad')
solution = ad.read_h5ad('output/datasets/joint_embedding/openproblems_bmmc_cite_phase1/openproblems_bmmc_cite_phase1.censor_dataset.output_solution.h5ad')

new_preds = metrics.create_anndata(solution, predictions.X)

print(metrics.calculate_metrics(new_preds, solution))