#!/bin/bash

# CMUMOSI with three random-seed runs
SEEDS="${SEEDS:-$RANDOM $RANDOM $RANDOM}"
for SEED in $SEEDS
do
  echo "========== CMUMOSI, seed=${SEED} =========="

  # A (audio only) – CM-DAE off as you used
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=100 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=a \
    --stage_epoch=50 \
    --gpu=0 \
    --lambda_m=0.0 \
    --use_cmdae=0

  # T (text)
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=100 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=t \
    --stage_epoch=50 \
    --gpu=0 \
    --lambda_m=0.1

  # V (video)
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=100 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=v \
    --stage_epoch=50 \
    --gpu=0 \
    --lambda_m=0.1

  # AT
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=100 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=at \
    --stage_epoch=50 \
    --gpu=0 \
    --lambda_m=0.1

  # AV
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=100 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=av \
    --stage_epoch=50 \
    --gpu=0 \
    --lambda_m=0.1

  # TV
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=100 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=tv \
    --stage_epoch=50 \
    --gpu=0 \
    --lambda_m=0.1

  # ATV
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=100 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=atv \
    --stage_epoch=50 \
    --gpu=0 \
    --lambda_m=0.1

done
