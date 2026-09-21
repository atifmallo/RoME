# RoME

Official implementation of **RoME: Robust Multimodal Expert Learning for Sentiment Analysis with Missing Modalities**, published in *Scientific Reports* (2026).

[[Paper]](https://doi.org/10.1038/s41598-026-60943-7)

RoME is a two-stage framework for multimodal sentiment analysis when audio, text, or visual inputs are missing. Stage 1 learns modality-specific representations using expert routing. Stage 2 builds robust multimodal representations through a Cross-Modal Interaction Transformer (CIT), Modality Reliability Re-Weighting (MRR), Residual Multimodal Corrector (RMC), and Cross-Modal Denoising Autoencoder (CM-DAE).

## Main contributions

- A two-stage expert framework that supports complete, partial, and single-modality inputs in one model.
- CIT for fine-grained interaction among available modalities.
- MRR for reliability-aware scaling of modality representations.
- RMC for residual refinement of the fused representation.
- CM-DAE for robust latent reconstruction under missing-modality conditions.

## Repository structure

```text
RoME/
├── README.md
├── config.py
├── requirements.txt
├── run_Rome_cmumosi.sh
├── run_Rome_cmumosei.sh
└── Rome/
    ├── train_Rome.py
    ├── model.py
    ├── utils.py
    ├── loss.py
    ├── dataloader_cmumosi.py
    └── modules/
        └── Attention_softmoe.py
```

## Environment

The code was tested with the following environment:

| Component | Version |
|---|---:|
| OS | Linux 6.8, x86_64 |
| Python | 3.8.20 |
| PyTorch | 1.12.0+cu116 |
| torchvision | 0.13.0+cu116 |
| CUDA used by PyTorch | 11.6 |
| cuDNN | 8.3.2 |
| GPU | NVIDIA GeForce RTX 3080, 10 GB |

Create the environment:

```bash
conda create -n rome python=3.8.20 -y
conda activate rome

pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

The NVIDIA driver may report a newer CUDA compatibility version. PyTorch 1.12.0 in the tested environment was compiled for CUDA 11.6.

## Datasets and features

Experiments use **CMU-MOSI** and **CMU-MOSEI**. Download the preprocessed archives below and comply with the original dataset licenses and terms.

| Dataset | Task | Preprocessed data |
|---|---|---|
| CMU-MOSI | Sentiment analysis | [Download](https://drive.google.com/file/d/1aJxArYfZsA-uLC0sOwIkjl_0ZWxiyPxj/view?usp=share_link) |
| CMU-MOSEI | Sentiment analysis | [Download](https://drive.google.com/file/d/1L6oDbtpFW2C4MwL5TQsEflY1WHjtv7L5/view?usp=share_link) |

RoME uses the following utterance-level features:

| Modality | Feature directory |
|---|---|
| Audio | `wav2vec-large-c-UTT` |
| Text | `deberta-large-4-UTT` |
| Visual | `manet_UTT` |

Arrange the extracted files as follows:

```text
dataset/
├── CMUMOSI/
│   ├── CMUMOSI_features_raw_2way.pkl
│   └── features/
│       ├── wav2vec-large-c-UTT/
│       ├── deberta-large-4-UTT/
│       └── manet_UTT/
└── CMUMOSEI/
    ├── CMUMOSEI_features_raw_2way.pkl
    └── features/
        ├── wav2vec-large-c-UTT/
        ├── deberta-large-4-UTT/
        └── manet_UTT/
```

The default data location is `./dataset`. To keep the data elsewhere, set its parent directory before training:

```bash
export ROME_DATA_ROOT=/absolute/path/to/dataset
```

Outputs are written to `./outputs` by default. An alternative location can be set with `ROME_OUTPUT_ROOT`.

## Training and evaluation

Run commands from the repository root.

### Quick test

This command trains one CMU-MOSI trimodal experiment. Replace `<RANDOM_SEED>` with an integer:

```bash
python -u Rome/train_Rome.py \
  --dataset=CMUMOSI \
  --audio-feature=wav2vec-large-c-UTT \
  --text-feature=deberta-large-4-UTT \
  --video-feature=manet_UTT \
  --seed=<RANDOM_SEED> \
  --batch-size=32 \
  --epochs=100 \
  --stage_epoch=50 \
  --lr=0.0001 \
  --hidden=256 \
  --depth=4 \
  --num_heads=2 \
  --drop_rate=0.5 \
  --attn_drop_rate=0.0 \
  --test_condition=atv \
  --gpu=0 \
  --lambda_m=0.1
```

### Reproduce all modality conditions

The scripts evaluate audio (`a`), text (`t`), visual (`v`), audio-text (`at`), audio-visual (`av`), text-visual (`tv`), and trimodal (`atv`) settings using three random seeds.

```bash
bash run_Rome_cmumosi.sh
bash run_Rome_cmumosei.sh
```

CMU-MOSI uses 100 epochs with a 50/50 split between Stage 1 and Stage 2. CMU-MOSEI uses 60 epochs with a 30/30 split. The released scripts use batch size 32, Adam, learning rate `1e-4`, hidden dimension 256, four layers, two attention heads, and dropout 0.5.

### Missing-modality conditions

Set `--test_condition` to one of:

```text
a, t, v, at, av, tv, atv
```

Binary masks simulate unavailable modalities during Stage 2. Stage 1 retains the complete unimodal inputs required to train the modality experts.

## Ablation study

The four Stage 2 components are enabled by default. They can be controlled with integer switches:

```bash
--use_cit=0
--use_mrr=0
--use_rmc=0
--use_cmdae=0
```

Adding `--do_switch_ablation` evaluates the best trained checkpoint with each component disabled in turn. This is the switchboard evaluation used to study CIT, MRR, RMC, and CM-DAE.

## Main results

Results are averaged over three independent runs with different random seeds. Each cell reports **Accuracy / F1-score (%)**.

### CMU-MOSI

| A | T | V | AV | AT | TV | Incomplete average | ATV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 65.82/64.21 | 90.30/90.25 | 73.84/73.26 | 73.84/73.26 | 91.14/91.14 | 91.99/91.92 | 81.16/80.67 | 91.98/91.94 |

### CMU-MOSEI

| A | T | V | AV | AT | TV | Incomplete average | ATV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 76.36/74.56 | 90.11/90.02 | 72.70/69.75 | 75.71/73.02 | 89.90/89.80 | 90.28/90.13 | 82.51/81.21 | 90.22/90.13 |

The incomplete average is computed over the six non-trimodal conditions: A, T, V, AV, AT, and TV.

## Outputs

Training logs, checkpoints, predictions, and JSON summaries are stored under the directory configured by `ROME_OUTPUT_ROOT`:

```text
outputs/
├── log/
├── model/
└── npz/
```

## Citation

If this code or method supports your research, please cite:

```bibtex
@article{hussain2026rome,
  title   = {Robust multimodal expert learning for sentiment analysis with missing modalities},
  author  = {Hussain, Atif and Gu, Yu and Islam, Iftekharul and Ali, Aamir and Zhang, He and Chilato, Chilato},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {10.1038/s41598-026-60943-7},
  url     = {https://doi.org/10.1038/s41598-026-60943-7}
}
```

## Acknowledgment

Thanks, [MoMKE](https://github.com/wxxv/MoMKE).
