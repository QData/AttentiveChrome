<<<<<<< HEAD
# DeepDiffChrome

[DeepDiff: Deep-learning for predicting Differential
gene expression from histone modifications](https://academic.oup.com/bioinformatics/article/34/17/i891/5093224)

```
@article{ArDeepDiff18,
author = {Sekhon, Arshdeep and Singh, Ritambhara and Qi, Yanjun},
title = {DeepDiff: DEEP-learning for predicting DIFFerential gene expression from histone modifications},
journal = {Bioinformatics},
volume = {34},
number = {17},
pages = {i891-i900},
year = {2018},
doi = {10.1093/bioinformatics/bty612},
URL = {http://dx.doi.org/10.1093/bioinformatics/bty612},
eprint = {/oup/backfile/content_public/journal/bioinformatics/34/17/10.1093_bioinformatics_bty612/2/bty612.pdf}
}
```
## Feature Generation

We used the five core histone modification (listed in the paper) read counts from REMC database as input matrix. We downloaded the files from REMC database. We converted 'tagalign.gz' format to 'bam' by using the command:   



gunzip <filename>.tagAlign.gz  
  
bedtools bedtobam -i <filename>.tagAlign -g hg19chrom.sizes > <filename>.bam   
  
Next, we used "bedtools multicov" to get the read counts. Bins of length 100 base-pairs (bp) are selected from regions (+/- 20000 bp) flanking the transcription start site (TSS) of each gene. The signal value of all five selected histone modifications from REMC in bins forms input matrix X, while log fold change in gene expression is the output y.   


For gene expression, we used the read count files available in REMC database and added 1 to all counts. 

We divided the genes into 3 separate sets for training(10,000 genes), validation(2360 genes) and testing(6100 genes). 

We performed training and validation on the first 2 sets and then reported Pearson Correlation Coefficient(PCC) scores of best performing epoch model for the third test data set. 

Sample dataset has been provided inside "data/" folder. For two cell types "Cell1" and "Cell2" under consideration, the expression value is in Cell1.expr.csv and Cell2.expr.csv for all genes. The first column is geneID, and the second column is expression value. The train, valid and test set inputs are in Cell*.train.csv, Cell*.valid.csv, and Cell*.test.csv. The columns represent: geneID_window,HM1,HM2,HM3,HM4,HM5. 

## Training Model
To train, validate and test the model for celltypes "Cell1" and "Cell2": 




&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python train.py --cell_1=Cell1 --model_name=attchrome --epochs=120 --lr=0.0001 --data_root=data/ --save_root=Results/



### Other Options
1. To specify DeepDiff variation: \
--model_name= \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;raw_d: difference of HMs \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;raw_c: concatenation of HMs \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;raw: raw features- difference and concatenation of HMs \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;raw_aux: raw features and auxiliary Cell type specific prediction features \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;aux: auxiliary Cell type specific prediction features \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;aux_siamese: auxiliary Cell type specific prediction features with siamese auxiliary \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;raw_aux_siamese: raw features and auxiliary Cell type specific prediction features with siamese auxiliary 

2. To save attention maps: \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;use option --save_attention_maps : saves Level II attention values in .txt file 

3. To change rnn size: \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--bin_rnn_size=32 




### Testing
To only test on a saved model: \
python train.py --test_on_saved_model --model_name=raw_d --data_root=data/ --save_root=Results/  
=======
# AttentiveChrome PyTorch

Coming Soon!
>>>>>>> a052baf48d76419e6805769e1fca8e3cb9b65ff0

