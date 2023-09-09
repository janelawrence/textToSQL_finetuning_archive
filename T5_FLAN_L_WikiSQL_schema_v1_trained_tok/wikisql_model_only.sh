#!/bin/bash

sudo ldconfig

export HOME=$PWD

# mkdir ./model_GPT2
# mkdir ./model_GPT2/wikisql

tar -xvf trained_tok.tar.gz

echo Ready to finetune
python wikisql_model_only.py


# tar -czf gpt2_wikisql_5epochs_model_only.tar.gz gpt2_wikisql_5epochs


