feature_dir=clip_feat

for DATASET in SoFAR
do
    python linear_probe.py \
    --dataset ${DATASET} \
    --feature_dir ${feature_dir} \
    --num_step 10 \
    --num_run 1
done
