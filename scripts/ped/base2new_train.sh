#!/bin/bash

#cd ../..
# custom config
DATA=PATH/TO/YOUR/DATASETS
TRAINER=PED

DATASET=$1
SHOTS=16
CFG=vit_b16_c2_ep20_batch4_4+4ctx

for SEED in 1 2 3
do
    DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done
python3 parse_test_res.py output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/

