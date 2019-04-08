# Attentive Chrome Kipoi

## Quick Start
### Creating new conda environtment using kipoi
`kipoi env create AttentiveChrome`

### Activating environment
`conda activate kipoi-AttentiveChrome`

## Command Line
### Getting example input file
`kipoi get-example AttentiveChrome/{model_name} -o example_file`

example: `kipoi get-example AttentiveChrome/E003 -o example_file`

### Predicting using example file 
`kipoi predict AttentiveChrome/{model_name} --dataloader_args='{"input_file": "example_file/input_file", "bin_size": 100}' -o example_predict.tsv`

