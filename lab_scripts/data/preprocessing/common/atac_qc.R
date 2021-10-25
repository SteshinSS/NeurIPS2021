#! /usr/bin/Rscript

suppressMessages(library(Signac))
suppressMessages(library(Seurat))
suppressMessages(library(GenomeInfoDb))
suppressMessages(library(EnsDb.Hsapiens.v75))
suppressMessages(library(biovizBase))
suppressMessages(library(ggplot2))
suppressMessages(library(patchwork))
suppressMessages(library(anndata))
suppressMessages(library(dplyr))
suppressMessages(library(readr))
suppressMessages(library(reticulate))
suppressMessages(library(Matrix))
suppressMessages(library(hdf5r))
suppressMessages(library(dplyr))
suppressMessages(library(tibble))

reticulate::use_condaenv("nips")
set.seed(1234);


QC_preprocessing <- function(counts_path, fragment_path) {
    # Input: path to HDF5 or AnnData peak-by-cell matrix; path to fragment_file
    # Output: Seurat object with @meta.data containing QC metrics
    
    # Read counts_path
    tryCatch(
    expr = { 
        counts <- Read10X_h5(counts_path)
    },
    error = function(e){
        print("Not HDF5 format, continuing with AnnData format")
    },
    warning = function(w){
        print("Not HDF5 format, continuing with AnnData format")
    },
    finally = {
        counts <- read_h5ad(counts_path)
    }
)

    # Load H5F5 or AnnData object and get counts 
    if (data.class(counts) == 'AnnDataR6') {
    counts <- t(counts$X)
    print('Data class is .h5ad matrix')
    } else if (data.class(counts) == 'dgCMatrix') {
    counts <- read_h5ad(counts)
    print('Data class is .h5 matrix')
    }
    
    # Create chromassay object: counts = peak-by-cell matrix; fragments = fragments file.
    # Note: fragments.tsv.gz index should be located in the same dir. If there is no index: 
    # gzip -d <fragments>.tsv.gz -> bgzip <fragments>.gz -> tabix -p bed <fragments>
    chrom_assay <- CreateChromatinAssay(
        counts = counts,
        sep = c(":", "-"),
        genome = 'hg19',
        fragments = fragment_path,
        min.cells = 0,
        min.features = 0)
    
    # Create SeuratObject from chromassay object
    seurat_object <- CreateSeuratObject(
        counts = chrom_assay,
        assay = "peaks")
    
    # Add gene annotations to the SeuratObject for the human genome:
    # extract gene annotations from EnsDb
    annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v75, verbose = FALSE)
    # change to UCSC style since the data was mapped to hg19
    seqlevelsStyle(annotations) <- 'UCSC'
    genome(annotations) <- "hg19"
    # add the gene information to the object
    Annotation(seurat_object) <- annotations
    
    
    # Count total fragments per cell barcode present in a fragment file
    total_fragments <- CountFragments(fragments = fragment_path)
    
    # Extract only common cells between peak-by-cell matrix and fragments file
    common_cells <- as.list(intersect(total_fragments$CB, row.names(seurat_object@meta.data)))

    # Filter only common cells from fragments file
    common_cells <- t(as.data.frame(common_cells))
    colnames(common_cells) <- 'CB'
    total_fragments_filtered <- merge(common_cells, total_fragments, by='CB')
    
    # Add total_fragments QC column to SeuratObject
    seurat_object@meta.data['fragments'] <- total_fragments_filtered["frequency_count"]
    
    # Calculate fraction of reads in peaks per cell
    seurat_object <- FRiP(object = seurat_object, assay = 'peaks', 
                    total.fragments = 'fragments', col.name = "FRiP")
    
    # Calculate TSSEnrichment score: ratio of fr-s centered at the TSS to fr-s in TSS-flanking regions
    seurat_object <- TSSEnrichment(object = seurat_object, fast = FALSE)
    
    # Calculate Nucleosome Signal: 
    seurat_object <- NucleosomeSignal(seurat_object, verbose = FALSE)
    
    # Create an object with blacklisted regions
    blaclkist <- as.data.frame(FractionCountsInRegion(
        object = seurat_object,
        assay = 'peaks',
        regions = blacklist_hg19))
    colnames(blaclkist) <- 'blacklist_fraction'
    
    # Calculate fraction of peaks blacklisted by the ENCODE consortium 
    seurat_object@meta.data["blacklist_fraction"] <- blaclkist["blacklist_fraction"]
    
    # Save QC attributes to df
    qc_table <- as.data.frame(seurat_object@meta.data)
    
    
    return(qc_table)  
}

# Block for running from command line
args = commandArgs(trailingOnly=TRUE)
counts = args[1]
fragment_path = args[2]
output_temp = args[3]
qc_table = QC_preprocessing(counts, fragment_path)
write.table(qc_table, output_temp, sep=';', row.names=TRUE)