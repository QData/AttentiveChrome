# epigenome data preprocessing
process to convert raw data from REMC/ENCODE BAM files to HM and gene expression 

**Step 1**: download.sh bam files for desired factor (needs ENCODE accession file number)

**Step 2**: get GeneFile.windows.bed(unzip GeneFile) indicating location of gene and name, we have gene expression for each gene [57epigenomes.RPKM.pc.gz](https://egg2.wustl.edu/roadmap/data/byDataType/rna/expression/)

**Step 3**:process.sh script uses "bedtools multicov" to get the read counts for bins of length 100 base-pairs (bp) are selected from regions (+/- 5000 bp) flanking the transcription start site (TSS) of each gene.

