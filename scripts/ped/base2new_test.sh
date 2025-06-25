#!/bin/bash

#cd ../..

# custom config
DATA=PATH/TO/YOUR/DATASETS
TRAINER=PED
DATASET=$1
SHOTS=16
CFG=vit_b16_c2_ep20_batch4_4+4ctx
LOADEP=20
SUB=new

for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
    DIR=output/base2new/test_${SUB}/${COMMON_DIR}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo "Evaluating model"
        echo "Runing the first phase job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
python3 parse_test_res.py output/base2new/test_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/