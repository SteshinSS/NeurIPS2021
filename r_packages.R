# See instructions here: https://www.notion.so/R-work-in-progress-b3fd6b10d895483f979306fbff4900b0

bioc_packages <- c(
  "SingleCellExperiment",
  "scran"
)

cran_packages <- c(
  "BiocManager",
  "dplyr",
  "Seurat",
  "anndata",
  "IRkernel",
  "Signac"
)

install.packages(cran_packages, Ncpus=8)
BiocManager::install(bioc_packages, Ncpus=8)

# Install R kernel for Jupyter
IRkernel::installspec()
