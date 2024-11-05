#!/usr/bin/env bash

set -x

PARTITION=$1
CONFIG=$2
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
QUOTATYPE=${QUOTATYPE:-'reserved'}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
MASTER_PORT=$RANDOM srun -p ${PARTITION} \
     --quotatype=${QUOTATYPE} \
     --gres=gpu:${GPUS_PER_NODE} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     --kill-on-bad-exit=1 \
     --preempt \
     ${SRUN_ARGS} \
     python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
