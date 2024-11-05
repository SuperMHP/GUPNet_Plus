#!/usr/bin/env bash
set -x
CONFIG=$1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$RANDOM}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

GPUS=${CUDA_VISIBLE_DEVICES//,/}
GPUS=${#GPUS}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:2}
