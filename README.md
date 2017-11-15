# AttentiveChrome

Reference Paper: [Attend and Predict: Using Deep Attention Model to Understand Gene Regulation by Selective Attention on Chromatin](https://arxiv.org/abs/1708.00339)

BibTex Citation:
```
@article{singh2017deepchrome,
  title={Attend and Predict: Understanding Gene Regulation by Selective Attention on Chromatin},
  author={Singh, Ritambhara and Lanchantin, Jack and Sekhon, Arshdeep  and Qi, Yanjun},
  journal={arXiv preprint arXiv:1708.00339},
  year={2017}
}
```

AttentiveChrome is a unified architecture to model and to interpret dependencies among chromatin factors for controlling gene regulation. AttentiveChrome uses a hierarchy of multiple Long short-term memory (LSTM) modules to encode the input signals and to model how various chromatin marks cooperate automatically. AttentiveChrome trains two levels of attention jointly with the target prediction, enabling it to attend differentially to relevant marks and to locate important positions per mark. We evaluate the model across 56 different cell types (tasks) in human. Not only is the proposed architecture more accurate, but its attention scores also provide a better interpretation than state-of-the-art feature visualization methods such as saliency map. 

**Feature Generation for AttentiveChrome model:** 

We used the five core histone modification (listed in the paper) read counts from REMC database as input matrix. We downloaded the files from [REMC dabase](http://egg2.wustl.edu/roadmap/web_portal/processed_data.html#ChipSeq_DNaseSeq). We converted 'tagalign.gz' format to 'bam' by using the command:
```
gunzip <filename>.tagAlign.gz
bedtools bedtobam -i <filename>.tagAlign -g hg19chrom.sizes > <filename>.bam 
```
Next, we used "bedtools multicov" to get the read counts. 
Bins of length 100 base-pairs (bp) are selected from regions (+/- 5000 bp) flanking the transcription start site (TSS) of each gene. The signal value of all five selected histone modifications from REMC in bins forms input matrix X, while discretized gene expression (label +1/-1) is the output y.

For gene expression, we used the RPKM read count files available in REMC database. We took the median of the RPKM read counts as threshold for assigning binary labels (-1: gene low, +1: gene high). 

We divided the genes into 3 separate sets for training, validation and testing. It was a simple file split resulting into 6601, 6601 and 6600 genes respectively. 

We performed training and validation on the first 2 sets and then reported AUC scores of best performing epoch model for the third test data set. 

Toy dataset has been provided inside "code/data" folder.

After downloading "code/" folder:

To perform training : 
```
th doall.lua
```
To perform testing/Get visualization output: 
```
the doall_eval.lua
```
