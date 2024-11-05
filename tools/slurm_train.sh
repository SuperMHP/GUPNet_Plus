#!/usr/bin/env bash

set -x

CONFIG=$1
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
QUOTATYPE=${QUOTATYPE:-'reserved'}
RUN_NUM=${RUN_NUM:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

for((i=1; i<=$RUN_NUM; i++));
do   
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
MASTER_PORT=$RANDOM srun -p ai4science \
     --quotatype=${QUOTATYPE} \
     --gres=gpu:${GPUS_PER_NODE} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     --kill-on-bad-exit=1 \
     --preempt \
     ${SRUN_ARGS} \
     python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
done
