

#!/bin/bash

# CMUMOSEI with three random-seed runs
# CMUMOSEI with three random-seed runs
for SEED in $SEEDS
do
  echo "========== CMUMOSEI, seed=${SEED} =========="

  # A
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSEI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=60 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=a \
    --stage_epoch=30 \
    --gpu=1 \
    --lambda_m=0.0 \
    --use_cmdae=0

  # T
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSEI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=60 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=t \
    --stage_epoch=30 \
    --gpu=1 \
    --lambda_m=0.1

  # V
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSEI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=60 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=v \
    --stage_epoch=30 \
    --gpu=1 \
    --lambda_m=0.1

  # AT
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSEI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=60 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=at \
    --stage_epoch=30 \
    --gpu=1 \
    --lambda_m=0.1

  # AV
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSEI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=60 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=av \
    --stage_epoch=30 \
    --gpu=1 \
    --lambda_m=0.1

  # TV
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSEI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=60 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=tv \
    --stage_epoch=30 \
    --gpu=1 \
    --lambda_m=0.1

  # ATV
  python -u Rome/train_Rome.py \
    --dataset=CMUMOSEI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=${SEED} \
    --batch-size=32 \
    --epochs=60 \
    --lr=0.0001 \
    --hidden=256 \
    --depth=4 \
    --num_heads=2 \
    --drop_rate=0.5 \
    --attn_drop_rate=0.0 \
    --test_condition=atv \
    --stage_epoch=30 \
    --gpu=1 \
    --lambda_m=0.1

done
