#!/bin/bash
set -ex
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}" 
export TPU_HOST="${TPU_HOST:-10.255.128.3}"
export TPU_NAME="${TPU_NAME:-tpu-v3-128-euw4a-4}"
export MODEL_DIR="${MODEL_DIR:-gs://fontgan_euw4/model_runs/fonts_128_1}"
#export DATASETS=gs://danbooru-euw4a/datasets/danbooru2019-s/danbooru2019-s-0*,gs://danbooru-euw4a/datasets/e621-s/e621-s-0*
#export DATASETS=gs://danbooru-euw4a/datasets/danbooru2019-s/danbooru2019-s-0*
export DATASETS=gs://fontgan_euw4/datasets/fonts_128/fonts_128-0*
export GIN_CONFIG="${GIN_CONFIG:-example_configs/biggan_font128.gin}"
exec python3 wrapper.py compare_gan/main.py --use_tpu --tfds_data_dir 'gs://danbooru-euw4a/tensorflow_datasets/' --model_dir "${MODEL_DIR}" --gin_config "$GIN_CONFIG"
