#!/bin/bash

sudo ldconfig

export HOME=$PWD

# mkdir ./model_GPT2
# mkdir ./model_GPT2/wikisql

echo Ready to finetune
python flan_t5_spider.py


# tar -czf gpt2_wikisql_5epochs_model_only.tar.gz gpt2_wikisql_5epochs