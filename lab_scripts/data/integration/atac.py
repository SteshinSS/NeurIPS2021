import re

import pybedtools


def parse_gtf(gtf_file, upstream):
    intervals = []
    with open(gtf_file) as f:
        for line in f:
            if not line.startswith("##"):
                line = line.rstrip("\n").split("\t")

                # Skip features out of chromosomes
                chromosome = line[0]
                if not chromosome.startswith("chr"):
                    continue

                # Skip not genes
                feature = line[2]
                if feature != "gene":
                    continue

                # Take gene_id (ENSG0000...)
                annotation = line[-1].split(";")[0]
                gene_id = re.search(r"gene_id \"(.+)\..+\"", annotation).groups(1)[0]

                # Substract 1 because gtf file starts intervals with 1
                start = int(line[3]) - 1
                end = int(line[4]) - 1

                strand = line[6]
                if strand == "-":
                    end += upstream
                else:
                    start = max(0, start - upstream)

                intervals.append((chromosome, start, end, feature, gene_id))
    return pybedtools.BedTool(intervals)


def parse_atac_adata(adata):
    intervals = []
    for line in adata.var.index.tolist():
        line = line.split("-")
        chromosome, start, end = line
        # Skip all regions out of chromosomes
        if not chromosome.startswith("chr"):
            continue
        start = int(start)
        end = int(end)
        intervals.append((chromosome, start, end))
    return pybedtools.BedTool(intervals)
