#!/usr/bin/env bash
set -x
CONFIG=$1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$RANDOM}
RUN_NUM=${RUN_NUM:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

GPUS=${CUDA_VISIBLE_DEVICES//,/}
GPUS=${#GPUS}

for((i=1; i<=$RUN_NUM; i++));
do   
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
done
