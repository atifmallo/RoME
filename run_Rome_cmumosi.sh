#!/bin/bash

# CMUMOSI with multi-seed runs: 62, 66, 70
for SEED in 62 66 70
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
    --use_cmdae=0 \
    --do_switch_ablation

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
    --lambda_m=0.1 \
    --do_switch_ablation

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
    --lambda_m=0.1 \
    --do_switch_ablation

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
    --lambda_m=0.1 \
    --do_switch_ablation

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
    --lambda_m=0.1 \
    --do_switch_ablation

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
    --lambda_m=0.1 \
    --do_switch_ablation

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
    --lambda_m=0.1 \
    --do_switch_ablation

done
