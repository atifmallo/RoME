# train_MoMKE.py  (MOSI/MOSEI ONLY)
# - CMUMOSI / CMUMOSEI only (all IEMOCAP code paths removed)
# - Two-stage training:
#     * Stage-1: unimodal regression (MSE or smooth L1 with --use_huber)
#     * Stage-2: new multimodal model (CIT + MRR + RMC or similar) trained directly on regression loss
#
# - Mask-aware, zero-excluded MOSI/MOSEI metrics (no inflated results)
# - Utterance-level evaluation pooling (default) via --eval_pool {utter,token}
# - Dev-split threshold selection for CMUMOSI (only if --save_dev_split > 0.0)
# - MOSEI keeps fixed threshold 0.0 (for legacy parity)
# - Optional CCC term for MOSI/MOSEI Stage-2 (--lambda_ccc)
# - Optimizer switch (adam | adamw), weight decay, cosine LR for Stage-2
# - Gradient clipping
# - Prediction dumps (--save_preds), from reloaded best epoch
# - Paper-style table across conditions (+ final JSON summary)
# - Safe device picking
# - FIX: keep original collate_fn when creating dev/test split
# - FIX: best-epoch selection/printing; recompute final metrics from reloaded best

import time
import datetime
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
import os
import warnings
import json
import sys
from typing import Optional

# --- make project root importable so we can do `import config` from parent dir ---
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from utils import Logger, get_loaders, build_model, generate_mask, generate_inputs
from loss import MaskedCELoss, MaskedMSELoss

warnings.filterwarnings("ignore")
import config


# ------------------------- alignment probe -------------------------
def _alignment_probe(args, loader, tag):
    """Quick sanity print of first batch: sample IDs and token counts per modality."""
    try:
        data = next(iter(loader))
    except StopIteration:
        print(f"[align/{tag}] loader empty")
        return
    audio_host, text_host, visual_host = data[0], data[1], data[2]
    audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
    qmask, umask, label = data[6], data[7], data[8]
    vidnames = list(data[-1])

    B = audio_host.shape[1]
    seqlen = audio_host.shape[0]
    matrix = generate_mask(seqlen, B, args.test_condition, first_stage=True)
    a_mask = np.reshape(matrix[0], (B, seqlen, 1))
    t_mask = np.reshape(matrix[1], (B, seqlen, 1))
    v_mask = np.reshape(matrix[2], (B, seqlen, 1))
    print(f"\n[align/{tag}] first-batch sanity (showing up to 8):")
    for i in range(min(8, B)):
        print(
            f"  #{i:02d} id={vidnames[i]}  aTok={int(a_mask[i].sum())}  "
            f"tTok={int(t_mask[i].sum())}  vTok={int(v_mask[i].sum())}  "
            f"umaskTok={int(umask[i].sum())}"
        )


# ------------------------- optimizer picker -------------------------
def _choose_optimizer(args, params):
    """Select Adam or AdamW with appropriate weight decay / L2."""
    opt = (args.optimizer or 'adam').lower()
    if opt == 'adamw':
        return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif opt == 'adam':
        return optim.Adam(params, lr=args.lr, weight_decay=args.l2)
    else:
        print(f"[warn] unknown --optimizer {args.optimizer}, falling back to Adam")
        return optim.Adam(params, lr=args.lr, weight_decay=args.l2)


# ------------------------- metric helpers (MOSI/MOSEI) -------------------------
def _mosi_mosei_binary_metrics_masked(
        labels: np.ndarray,
        preds_cont: np.ndarray,
        masks: np.ndarray,
        thresh: float = 0.0,
        exclude_zero: bool = True
):
    """Binary metrics for MOSI/MOSEI from continuous labels/preds."""
    labels = labels.squeeze()
    preds = preds_cont.squeeze()
    assert labels.shape == preds.shape == masks.shape, "labels/preds/masks must have same flattened length"

    idx = masks.astype(bool)
    if exclude_zero:
        idx = idx & (labels != 0)

    if idx.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0

    l = labels[idx]
    p = preds[idx]

    mae = float(np.mean(np.abs(l - p)))
    corr = float(np.corrcoef(l, p)[0][1]) if l.size > 1 else 0.0

    y_true = (l > 0.0)
    y_pred = (p > thresh)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return mae, corr, float(acc), float(f1)


def _sweep_threshold_masked(labels: np.ndarray, preds_cont: np.ndarray, masks: np.ndarray, on: str = 'f1'):
    """Sweep ONLY the prediction threshold; labels fixed at 0.0 (MOSI dev)."""
    on = on.lower()
    assert on in ('f1', 'acc')
    ts = np.linspace(-2.0, 2.0, num=401)
    best_t, best_v, best_metrics = 0.0, -1e18, {}
    for t in ts:
        mae, corr, acc, f1 = _mosi_mosei_binary_metrics_masked(labels, preds_cont, masks, thresh=t, exclude_zero=True)
        score = f1 if on == 'f1' else acc
        if score > best_v:
            best_v = score
            best_t = float(t)
            best_metrics = dict(mae=mae, corr=corr, acc=acc, f1=f1)
    return best_t, best_metrics


# ------------------------- loss helpers -------------------------
def _masked_ccc_loss(preds: torch.Tensor, labels: torch.Tensor, umask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Masked Concordance Correlation Coefficient loss = 1 - CCC."""
    m = umask.view(-1).float()
    if m.sum() < 1:
        return preds.new_tensor(0.0)
    x = preds.view(-1)
    y = labels.view(-1).float()
    mx = (m * x).sum() / (m.sum() + eps)
    my = (m * y).sum() / (m.sum() + eps)
    vx = (m * (x - mx) ** 2).sum() / (m.sum() + eps)
    vy = (m * (y - my) ** 2).sum() / (m.sum() + eps)
    cov = (m * (x - mx) * (y - my)).sum() / (m.sum() + eps)
    ccc = 2 * cov / (vx + vy + (mx - my) ** 2 + eps)
    return 1.0 - ccc


# ------------------------- pooling helpers -------------------------
def _ensure_BxL(x, B, L):
    """Try to reshape/permute tensor x into [B, L] if it's [L, B] or flat."""
    if x.dim() == 2:
        if x.size(0) == L and x.size(1) == B:
            return x.transpose(0, 1).contiguous()  # -> [B, L]
        elif x.size(0) == B and x.size(1) == L:
            return x
    return x.view(B, L)


def _pool_batch_utter(preds_BLC, labels_BL, umask_BL, n_classes):
    """Average across valid timesteps per utterance (regression)."""
    B, L = umask_BL.size()
    m = umask_BL.float()
    msum = m.sum(dim=1).clamp(min=1e-8)

    if n_classes == 1:
        cont = (preds_BLC.squeeze(-1) * m).sum(dim=1) / msum
        labs = (labels_BL.float() * m).sum(dim=1) / msum
        return cont.detach().cpu().numpy(), labs.detach().cpu().numpy(), np.ones(B, dtype=np.float32)
    else:
        # classification path (unused here, kept for completeness)
        probs = torch.softmax(preds_BLC, dim=-1)
        pbar = (probs * m.unsqueeze(-1)).sum(dim=1) / msum.unsqueeze(-1)
        pred_cls = torch.argmax(pbar, dim=-1)
        first_idx = (m > 0).float().argmax(dim=1)
        idx = torch.arange(B, device=labels_BL.device)
        labs = labels_BL[idx, first_idx]
        return pred_cls.detach().cpu().numpy(), labs.detach().cpu().numpy(), np.ones(B, dtype=np.float32)


# ------------------------- core train/eval -------------------------
def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader,
                        optimizer=None, train=False, first_stage=True,
                        epoch_norm: float = 1.0, collect_for_thresh: bool = False):
    """Run one epoch over `dataloader` (Stage-1 or Stage-2)."""
    preds, preds_a, preds_t, preds_v, masks, labels = [], [], [], [], [], []
    losses = []

    # utter-level accumulators for Stage-2 when eval_pool=utter
    utt_preds_cont, utt_labels, utt_masks = [], [], []

    stash_labels, stash_preds_cont, stash_masks, stash_vids = [], [], [], []

    cuda = (args.device.type == 'cuda')
    assert (not train) or (optimizer is not None)
    model.train() if train else model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        qmask, umask, label = data[6], data[7], data[8]
        vidnames = list(data[-1])

        seqlen = audio_host.size(0)
        batch = audio_host.size(1)

        # Missingness curriculum: random modality drop for host/guest
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage)
        audio_host_mask = torch.LongTensor(np.reshape(matrix[0], (batch, seqlen, 1)).transpose(1, 0, 2))
        text_host_mask = torch.LongTensor(np.reshape(matrix[1], (batch, seqlen, 1)).transpose(1, 0, 2))
        visual_host_mask = torch.LongTensor(np.reshape(matrix[2], (batch, seqlen, 1)).transpose(1, 0, 2))

        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage)
        audio_guest_mask = torch.LongTensor(np.reshape(matrix[0], (batch, seqlen, 1)).transpose(1, 0, 2))
        text_guest_mask = torch.LongTensor(np.reshape(matrix[1], (batch, seqlen, 1)).transpose(1, 0, 2))
        visual_guest_mask = torch.LongTensor(np.reshape(matrix[2], (batch, seqlen, 1)).transpose(1, 0, 2))

        masked_audio_host = audio_host * audio_host_mask
        masked_audio_guest = audio_guest * audio_guest_mask
        masked_text_host = text_host * text_host_mask
        masked_text_guest = text_guest * text_guest_mask
        masked_visual_host = visual_host * visual_host_mask
        masked_visual_guest = visual_guest * visual_guest_mask

        if cuda:
            dev = args.device
            masked_audio_host, audio_host_mask = masked_audio_host.to(dev), audio_host_mask.to(dev)
            masked_text_host, text_host_mask = masked_text_host.to(dev), text_host_mask.to(dev)
            masked_visual_host, visual_host_mask = masked_visual_host.to(dev), visual_host_mask.to(dev)
            masked_audio_guest, audio_guest_mask = masked_audio_guest.to(dev), audio_guest_mask.to(dev)
            masked_text_guest, text_guest_mask = masked_text_guest.to(dev), text_guest_mask.to(dev)
            masked_visual_guest, visual_guest_mask = masked_visual_guest.to(dev), visual_guest_mask.to(dev)
            qmask = qmask.to(dev)
            umask = umask.to(dev)
            label = label.to(dev)

        masked_input_features = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host,
                                                masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
        input_features_mask = generate_inputs(audio_host_mask, text_host_mask, visual_host_mask,
                                              audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)

        with torch.set_grad_enabled(train):
            # New model forward signature: no 'targets' or 'epoch' kwargs.
            hidden, out, out_a, out_t, out_v, weight_save = model(
                masked_input_features[0],
                input_features_mask[0],
                umask,
                first_stage=first_stage
            )

            lp_ = out.view(-1, out.size(2))  # N x 1 (regression)
            lp_a = out_a.view(-1, out_a.size(2))
            lp_t = out_t.view(-1, out_t.size(2))
            lp_v = out_v.view(-1, out_v.size(2))
            labels_ = label.view(-1)

            # ---------------- MOSI/MOSEI: regression only ----------------
            if first_stage:
                # Stage-1: unimodal regression on A/T/V heads (same as original)
                if args.use_huber:
                    loss_a = torch.nn.functional.smooth_l1_loss(
                        lp_a.squeeze(-1), labels_.float(), reduction='none'
                    )
                    loss_t = torch.nn.functional.smooth_l1_loss(
                        lp_t.squeeze(-1), labels_.float(), reduction='none'
                    )
                    loss_v = torch.nn.functional.smooth_l1_loss(
                        lp_v.squeeze(-1), labels_.float(), reduction='none'
                    )
                    mask1d = umask.view(-1)
                    denom = (mask1d.sum() + 1e-8)
                    loss_a = (loss_a * mask1d).sum() / denom
                    loss_t = (loss_t * mask1d).sum() / denom
                    loss_v = (loss_v * mask1d).sum() / denom
                else:
                    loss_a = reg_loss(lp_a, labels_, umask)
                    loss_t = reg_loss(lp_t, labels_, umask)
                    loss_v = reg_loss(lp_v, labels_, umask)
                loss = (loss_a + loss_t + loss_v) / 3.0
            else:
                # Stage-2: train on the final fused prediction 'out' from new multimodal head
                preds_cont = lp_.squeeze(-1)  # [N]
                if args.use_huber:
                    loss_flat = torch.nn.functional.smooth_l1_loss(
                        preds_cont, labels_.float(), reduction='none'
                    )
                    mask1d = umask.view(-1)
                    denom = (mask1d.sum() + 1e-8)
                    loss = (loss_flat * mask1d).sum() / denom
                else:
                    loss = reg_loss(lp_, labels_, umask)

                # optional CCC term for MOSI/MOSEI (on Stage-2 only)
                if getattr(args, 'lambda_ccc', 0.0) > 0.0:
                    ccc_loss = _masked_ccc_loss(preds_cont, labels_.float(), umask.view(-1))
                    loss = loss + float(args.lambda_ccc) * ccc_loss

                # optional CCC term for MOSI/MOSEI (on Stage-2 only)
            if getattr(args, 'lambda_ccc', 0.0) > 0.0:
                ccc_loss = _masked_ccc_loss(preds_cont, labels_.float(), umask.view(-1))
                loss = loss + float(args.lambda_ccc) * ccc_loss

            # ---------------- CM-DAE reconstruction loss (Stage-2 only) ----------------
            # We reuse args.lambda_m as the CM-DAE weight. Loss is only added in training mode.
            if train and getattr(args, 'lambda_m', 0.0) > 0.0:
                cache = getattr(model, 'cmdae_cache', None)
                if cache is not None:
                    z_a = cache["z_a"]
                    z_t = cache["z_t"]
                    z_v = cache["z_v"]
                    z_hat_a = cache["z_hat_a"]
                    z_hat_t = cache["z_hat_t"]
                    z_hat_v = cache["z_hat_v"]

                    # simple L2 recon loss averaged over batch and modalities
                    dae_loss_a = torch.mean((z_hat_a - z_a) ** 2)
                    dae_loss_t = torch.mean((z_hat_t - z_t) ** 2)
                    dae_loss_v = torch.mean((z_hat_v - z_v) ** 2)
                    dae_loss = (dae_loss_a + dae_loss_t + dae_loss_v) / 3.0

                    loss = loss + float(args.lambda_m) * dae_loss
            # ---------------------------------------------------------------------------

            if train:
                loss.backward()
                if args.grad_clip is not None and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            preds_a.append(lp_a.data.cpu().numpy())
            preds_t.append(lp_t.data.cpu().numpy())
            preds_v.append(lp_v.data.cpu().numpy())
            preds.append(lp_.data.cpu().numpy())
            labels.append(labels_.data.cpu().numpy())
            masks.append(umask.view(-1).cpu().numpy())
            losses.append(loss.item())

            # ----- stash & utter pooling for Stage-2 -----
            if not first_stage:
                B = masked_input_features[0].size(1)
                L = masked_input_features[0].size(0)
                out_BLC = out
                lab_BL = _ensure_BxL(label, B, L)
                uma_BL = _ensure_BxL(umask, B, L)

                if args.eval_pool == 'utter':
                    pu, lu, mu = _pool_batch_utter(out_BLC, lab_BL, uma_BL, args.n_classes)
                    utt_preds_cont.append(pu)
                    utt_labels.append(lu)
                    utt_masks.append(mu)

                if collect_for_thresh:
                    if args.eval_pool == 'utter':
                        stash_preds_cont.append(pu)
                        stash_labels.append(lu)
                        stash_masks.append(mu)
                        stash_vids.extend(vidnames)
                    else:
                        stash_preds_cont.append(lp_.data.cpu().numpy().squeeze(-1))
                        stash_labels.append(labels_.data.cpu().numpy())
                        stash_masks.append(umask.view(-1).cpu().numpy())
                        stash_vids.extend(vidnames)

    assert preds != [], 'Error: dataloader yielded no batches'
    preds = np.concatenate(preds)
    preds_a = np.concatenate(preds_a)
    preds_t = np.concatenate(preds_t)
    preds_v = np.concatenate(preds_v)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)

    avg_loss = float(np.mean(losses)) if len(losses) else 0.0
    metrics = dict(mae=0.0, corr=0.0, acc=0.0, f1=0.0, total_loss=avg_loss)

    # ---- MOSI/MOSEI (regression) ----
    if (not first_stage) and (args.eval_pool == 'utter') and len(utt_labels) > 0:
        u_pred = np.concatenate(utt_preds_cont)
        u_lab = np.concatenate(utt_labels)
        u_m = np.concatenate(utt_masks)
        mae, corr, acc, f1 = _mosi_mosei_binary_metrics_masked(u_lab, u_pred, u_m, thresh=0.0, exclude_zero=True)
        metrics.update(dict(mae=mae, corr=corr, acc=acc, f1=f1))
        stash = dict(labels=np.array([]), preds_cont=np.array([]), masks=np.array([]), vids=np.array([]))
        if collect_for_thresh:
            stash = dict(
                labels=np.concatenate(stash_labels) if len(stash_labels) else np.array([]),
                preds_cont=np.concatenate(stash_preds_cont) if len(stash_preds_cont) else np.array([]),
                masks=np.concatenate(stash_masks) if len(stash_masks) else np.array([]),
                vids=np.array(stash_vids)
            )
        return metrics, {}, stash  # aux_means empty

    # legacy token-level (flattened) for MOSI/MOSEI
    cont = preds.squeeze(-1)
    mae, corr, acc, f1 = _mosi_mosei_binary_metrics_masked(labels, cont, masks, thresh=0.0, exclude_zero=True)
    metrics.update(dict(mae=mae, corr=corr, acc=acc, f1=f1))

    stash = dict(labels=np.array([]), preds_cont=np.array([]),
                 masks=np.array([]),
                 vids=np.array(stash_vids) if len(stash_vids) else np.array([]))
    if collect_for_thresh:
        stash = dict(
            labels=np.concatenate(stash_labels) if len(stash_labels) else np.array([]),
            preds_cont=np.concatenate(stash_preds_cont) if len(stash_preds_cont) else np.array([]),
            masks=np.concatenate(stash_masks) if len(stash_masks) else np.array([]),
            vids=np.array(stash_vids)
        )

    return metrics, {}, stash  # aux_means empty


# ------------------------- run one condition -------------------------
def run_one_condition(args, train_loaders, test_loaders, adim, tdim, vdim, outdir_preds: Optional[str] = None):
    """Train/test one condition (e.g. a/t/v/av/at/tv/atv) for MOSI/MOSEI."""
    print('====== Training and Testing =======')
    folder_mae, folder_corr, folder_acc, folder_f1 = [], [], [], []
    final_best_state_dict = None

    # NEW: per-condition ablation results (ALL_ON, CIT_OFF, MRR_OFF, RMC_OFF, DAE_OFF)
    ablation_results_all_folds = {}

    assert args.dataset in ['CMUMOSI', 'CMUMOSEI'], "This trainer now supports only CMUMOSI/CMUMOSEI."

    # Only MOSI uses a dev split for threshold sweeps by default
    use_dev = (args.dataset == 'CMUMOSI' and args.save_dev_split > 0.0)

    for ii in range(args.num_folder):
        print(f'>>>>> Cross-validation: training on the {ii + 1} folder >>>>>')

        train_loader = train_loaders[ii]
        test_loader = test_loaders[ii]

        # ---- DEV SPLIT (preserve collate_fn!) ----
        dev_loader = None
        if use_dev:
            N = len(test_loader.dataset)
            K = max(1, int(round(N * float(args.save_dev_split))))
            from torch.utils.data import Subset, DataLoader
            base_collate = getattr(test_loader, 'collate_fn', None)

            dev_subset = Subset(test_loader.dataset, list(range(K)))
            test_subset = Subset(test_loader.dataset, list(range(K, N)))

            dev_loader = DataLoader(
                dev_subset,
                batch_size=test_loader.batch_size,
                shuffle=False,
                num_workers=test_loader.num_workers,
                drop_last=False,
                collate_fn=base_collate,
                pin_memory=getattr(test_loader, 'pin_memory', False)
            )
            test_loader = DataLoader(
                test_subset,
                batch_size=test_loader.batch_size,
                shuffle=False,
                num_workers=test_loader.num_workers,
                drop_last=False,
                collate_fn=base_collate,
                pin_memory=getattr(test_loader, 'pin_memory', False)
            )
            print(f"[dev] Using {K}/{N} samples as dev for thresholding.")

        start_time = time.time()

        print('-' * 80)
        model = build_model(args, adim, tdim, vdim)
        if args.device.type == 'cuda':
            model.to(args.device)

        optimizer = _choose_optimizer(args, model.parameters())
        scheduler = None  # created later for Stage-2 if --cosine_stage2

        reg_loss = MaskedMSELoss()
        cls_loss = MaskedCELoss()  # unused in regression mode

        print('-' * 80)
        print('Step2: training (multiple epochs)')

        train_acc_as, train_acc_ts, train_acc_vs = [], [], []
        start_first_stage_time = time.time()

        stage1_epochs = int(args.stage_epoch)
        stage2_epochs = max(0, int(args.epochs) - stage1_epochs)
        print("------- Starting the first stage! -------")

        # -------- Stage-1: unimodal regression --------
        models_stage1_sd = []
        for epoch in range(stage1_epochs):
            denom = max(1, stage1_epochs)
            epoch_norm = min(1.0, (epoch + 1) / float(denom))

            train_m, train_aux, _ = train_or_eval_model(
                args, model, reg_loss, cls_loss, train_loader,
                optimizer=optimizer, train=True, first_stage=True,
                epoch_norm=epoch_norm, collect_for_thresh=False
            )

            test_m, test_aux, _ = train_or_eval_model(
                args, model, reg_loss, cls_loss, test_loader,
                optimizer=None, train=False, first_stage=True,
                epoch_norm=1.0, collect_for_thresh=False
            )

            train_acc_atv = [train_m.get('acc', 0.0)] * 3
            test_acc_atv = [test_m.get('acc', 0.0)] * 3

            print(
                f'epoch:{epoch}; a_acc_train:{train_acc_atv[0]:.3f}; '
                f't_acc_train:{train_acc_atv[1]:.3f}; v_acc_train:{train_acc_atv[2]:.3f}'
            )
            print(
                f'epoch:{epoch}; a_acc_test:{test_acc_atv[0]:.3f}; '
                f't_acc_test:{test_acc_atv[1]:.3f}; v_acc_test:{test_acc_atv[2]:.3f}'
            )
            print('----------')

            models_stage1_sd.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            train_acc_as.append(train_acc_atv[0])
            train_acc_ts.append(train_acc_atv[1])
            train_acc_vs.append(train_acc_atv[2])

        end_first_stage_time = time.time()

        # [BEST TRANSFORMER COPY] After Stage-1, copy best A/T/V Transformer weights into the model for Stage-2.
        # NOTE: In your new model, there may be no modality-specific transformer keys; then this is effectively a no-op.
        if stage1_epochs > 0 and (not args.no_stage_copy):
            model_idx_a = int(torch.argmax(torch.Tensor(train_acc_as)))
            model_idx_t = int(torch.argmax(torch.Tensor(train_acc_ts)))
            model_idx_v = int(torch.argmax(torch.Tensor(train_acc_vs)))

            print(f'best_epoch_a: {model_idx_a}')
            print(f'best_epoch_t: {model_idx_t}')
            print(f'best_epoch_v: {model_idx_v}')

            state_now = model.state_dict()
            snap_a = models_stage1_sd[model_idx_a]
            snap_t = models_stage1_sd[model_idx_t]
            snap_v = models_stage1_sd[model_idx_v]
            for k in list(state_now.keys()):
                if 'Transformer' in k:
                    if 'a' in k:
                        state_now[k] = snap_a.get(k, state_now[k])
                    if 't' in k:
                        state_now[k] = snap_t.get(k, state_now[k])
                    if 'v' in k:
                        state_now[k] = snap_v.get(k, state_now[k])
            model.load_state_dict(state_now, strict=False)
            print("[Stage-1 copy] loaded Transformer weights from best A/T/V epochs")
            print("------- Starting the second stage! -------")

        # Cosine LR for Stage-2
        if args.cosine_stage2 and stage2_epochs > 0:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage2_epochs, eta_min=args.lr * 0.1
            )

        best_epoch_abs = None
        best_sel_val = -1e18
        best_state = None

        stage1_state_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # -------- Stage-2: multimodal contributions --------
        start_second_stage_time = time.time()
        for i in range(stage2_epochs):
            epoch = stage1_epochs + i
            denom = max(1, stage2_epochs)
            epoch_norm = (i + 1) / float(denom)
            epoch_norm = max(0.0, min(1.0, epoch_norm))

            train_m, train_aux, _ = train_or_eval_model(
                args, model, reg_loss, cls_loss, train_loader,
                optimizer=optimizer, train=True, first_stage=False,
                epoch_norm=epoch_norm, collect_for_thresh=False
            )

            collect_flag = True  # MOSI/MOSEI only

            test_m, test_aux, test_stash = train_or_eval_model(
                args, model, reg_loss, cls_loss, test_loader,
                optimizer=None, train=False, first_stage=False,
                epoch_norm=1.0, collect_for_thresh=collect_flag
            )

            # per-epoch dev-thresholding (MOSI only)
            if use_dev and (args.dataset == 'CMUMOSI'):
                dev_m, _, dev_stash = train_or_eval_model(
                    args, model, reg_loss, cls_loss, dev_loader,
                    optimizer=None, train=False, first_stage=False,
                    epoch_norm=1.0, collect_for_thresh=True
                )
                if dev_stash['labels'].size > 0 and test_stash['labels'].size > 0:
                    selected_threshold, _ = _sweep_threshold_masked(
                        dev_stash['labels'], dev_stash['preds_cont'], dev_stash['masks'],
                        on=('f1' if args.select_by == 'f1' else 'acc')
                    )
                    mae, corr, acc, f1 = _mosi_mosei_binary_metrics_masked(
                        test_stash['labels'], test_stash['preds_cont'], test_stash['masks'],
                        thresh=selected_threshold, exclude_zero=True
                    )
                    test_m = dict(mae=mae, corr=corr, acc=acc, f1=f1, total_loss=test_m.get('total_loss', 0.0))

            # selection key (ACC/F1 only)
            if args.select_by == 'f1':
                sel = test_m['f1']
            else:
                sel = test_m['acc']

            if sel > best_sel_val:
                best_sel_val = sel
                best_epoch_abs = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if scheduler is not None:
                scheduler.step()

            print(
                f'epoch:{epoch}; '
                f'train_mae_{args.test_condition}:{train_m["mae"]:.3f}; '
                f'train_corr_{args.test_condition}:{train_m["corr"]:.3f}; '
                f'train_fscore_{args.test_condition}:{train_m["f1"]:2.2%}; '
                f'train_acc_{args.test_condition}:{train_m["acc"]:2.2%}; '
                f'train_loss_{args.test_condition}:{train_m["total_loss"]:.4f}'
            )
            print(
                f'epoch:{epoch}; '
                f'test_mae_{args.test_condition}:{test_m["mae"]:.3f}; '
                f'test_corr_{args.test_condition}:{test_m["corr"]:.3f}; '
                f'test_fscore_{args.test_condition}:{test_m["f1"]:2.2%}; '
                f'test_acc_{args.test_condition}:{test_m["acc"]:2.2%}; '
                f'test_loss_{args.test_condition}:{test_m["total_loss"]:.4f}'
            )
            print('----------')

        end_second_stage_time = time.time()
        print("-" * 80)
        print(f"Time of first stage: {end_first_stage_time - start_time:.6f}s")
        print(f"Time of second stage: {end_second_stage_time - end_first_stage_time:.6f}s")
        print("-" * 80)

        print(f'Step3: saving and testing on the {ii + 1} folder')

        # ---- FINAL: reload best checkpoint and recompute final metrics/preds ----
        if stage2_epochs == 0:
            best_epoch_abs = stage1_epochs - 1
            best_state = stage1_state_snapshot

        if best_state is not None:
            try:
                model.load_state_dict({k: v.to(args.device) for k, v in best_state.items()}, strict=False)
            except Exception:
                pass

        # ----- Final evaluation: MOSI/MOSEI -----
        selected_threshold = 0.0

        # Only MOSI uses dev-swept threshold. MOSEI keeps fixed 0.0 for exact legacy parity.
        if use_dev and (args.dataset == 'CMUMOSI'):
            dev_m, _, dev_stash = train_or_eval_model(
                args, model, MaskedMSELoss(), MaskedCELoss(), dev_loader,
                optimizer=None, train=False, first_stage=False,
                epoch_norm=1.0, collect_for_thresh=True
            )
            if dev_stash['labels'].size > 0:
                selected_threshold, _ = _sweep_threshold_masked(
                    dev_stash['labels'], dev_stash['preds_cont'], dev_stash['masks'],
                    on=('f1' if args.select_by == 'f1' else 'acc')
                )

        final_test_m, _, final_test_stash = train_or_eval_model(
            args, model, MaskedMSELoss(), MaskedCELoss(), test_loader,
            optimizer=None, train=False, first_stage=False,
            epoch_norm=1.0, collect_for_thresh=True
        )
        if final_test_stash['labels'].size > 0:
            mae, corr, acc, f1 = _mosi_mosei_binary_metrics_masked(
                final_test_stash['labels'], final_test_stash['preds_cont'], final_test_stash['masks'],
                thresh=selected_threshold, exclude_zero=True
            )
            final_test_m = dict(mae=mae, corr=corr, acc=acc, f1=f1, total_loss=final_test_m.get('total_loss', 0.0))

        # NEW: store ALL_ON as baseline in ablation dict for this fold/condition
        ablation_results_this_fold = {}
        ablation_results_this_fold['ALL_ON'] = dict(final_test_m)

        # ----------------- NEW: switchboard ablations (test-time only) -----------------
        if getattr(args, 'do_switch_ablation', False) and (best_state is not None):
            print(
                "\n[ABLATION] Running switchboard ablations on best checkpoint "
                f"for condition {args.test_condition}"
            )

            # Save a clean CPU copy of best weights
            base_state = {k: v.detach().cpu().clone() for k, v in best_state.items()}

            def _eval_with_switches(tag, use_cit, use_mrr, use_rmc, use_cmdae):
                # Reload identical weights
                try:
                    model.load_state_dict({k: v.to(args.device) for k, v in base_state.items()}, strict=False)
                except Exception:
                    pass

                # Flip switches in the model (if present)
                if hasattr(model, 'use_cit'):
                    model.use_cit = bool(use_cit)
                if hasattr(model, 'use_mrr'):
                    model.use_mrr = bool(use_mrr)
                if hasattr(model, 'use_rmc'):
                    model.use_rmc = bool(use_rmc)
                if hasattr(model, 'use_cmdae'):
                    model.use_cmdae = bool(use_cmdae)

                m, _, stash = train_or_eval_model(
                    args, model, MaskedMSELoss(), MaskedCELoss(), test_loader,
                    optimizer=None, train=False, first_stage=False,
                    epoch_norm=1.0, collect_for_thresh=True
                )
                if stash['labels'].size > 0:
                    mae_, corr_, acc_, f1_ = _mosi_mosei_binary_metrics_masked(
                        stash['labels'], stash['preds_cont'], stash['masks'],
                        thresh=selected_threshold, exclude_zero=True
                    )
                    m = dict(mae=mae_, corr=corr_, acc=acc_, f1=f1_, total_loss=m.get('total_loss', 0.0))

                print(
                    f"[ABLATION/{args.test_condition}/{tag}] "
                    f"mae={m['mae']:.3f} corr={m['corr']:.3f} "
                    f"acc={m['acc'] * 100:5.2f} f1={m['f1'] * 100:5.2f}"
                )
                ablation_results_this_fold[tag] = dict(m)  # NEW: record this ablation
                return m

            # 0) ALL ON (just print the already computed final_test_m)
            print(
                f"[ABLATION/{args.test_condition}/ALL_ON] "
                f"mae={final_test_m['mae']:.3f} corr={final_test_m['corr']:.3f} "
                f"acc={final_test_m['acc'] * 100:5.2f} f1={final_test_m['f1'] * 100:5.2f}"
            )

            # 1) CIT OFF  (MRR, RMC, DAE ON)
            _eval_with_switches("CIT_OFF", False, True, True, True)
            # 2) MRR OFF  (CIT, RMC, DAE ON)
            _eval_with_switches("MRR_OFF", True, False, True, True)
            # 3) RMC OFF  (CIT, MRR, DAE ON)
            _eval_with_switches("RMC_OFF", True, True, False, True)
            # 4) DAE OFF  (CIT, MRR, RMC ON)
            _eval_with_switches("DAE_OFF", True, True, True, False)
        # ------------------------------------------------------------------------------

        # merge ablation results for this fold into global dict
        # (for now we just keep the last fold since num_folder=1 in your setup)
        ablation_results_all_folds = ablation_results_this_fold

        if outdir_preds and args.save_preds:
            os.makedirs(outdir_preds, exist_ok=True)
            dump_path = os.path.join(outdir_preds, f"preds_fold{ii}_bestepoch{best_epoch_abs}.npz")
            np.savez_compressed(
                dump_path,
                labels=final_test_stash.get('labels', np.array([])),
                preds_cont=final_test_stash.get('preds_cont', np.array([])),
                masks=final_test_stash.get('masks', np.array([])),
                vids=final_test_stash.get('vids', np.array([])),
                best_threshold=(selected_threshold if (use_dev and args.dataset == 'CMUMOSI') else 0.0)
            )
            print(f"[preds] Wrote: {dump_path}")

        folder_mae.append(final_test_m['mae'])
        folder_corr.append(final_test_m['corr'])
        folder_f1.append(final_test_m['f1'])
        folder_acc.append(final_test_m['acc'])

        print(
            f"The best({args.select_by}) epoch of test_condition ({args.test_condition}): {best_epoch_abs} "
            f"--test_mae {final_test_m['mae']} --test_corr {final_test_m['corr']} "
            f"--test_fscores {final_test_m['f1']} --test_acc {final_test_m['acc']}."
        )

        final_best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        end_time = time.time()
        print(f'>>>>> Finish: training on the {ii + 1} folder, duration: {end_time - start_time} >>>>>')

    print('-' * 80)
    print(
        f"Folder avg: test_condition ({args.test_condition}) "
        f"--test_mae {np.mean(folder_mae)} "
        f"--test_corr {np.mean(folder_corr)} "
        f"--test_fscores {np.mean(folder_f1)} "
        f"--test_acc{np.mean(folder_acc)}"
    )
    return {
        'ACC': float(np.mean(folder_acc)),
        'F1': float(np.mean(folder_f1)),
        'MAE': float(np.mean(folder_mae)),
        'CORR': float(np.mean(folder_corr))
    }, final_best_state_dict, ablation_results_all_folds


# ------------------------- table utilities -------------------------
def _fmt_pair(v1, v2):
    return f"{v1 * 100:.2f}/{v2 * 100:.2f}"


def load_paper_rows(dataset):
    base = os.path.join('paper_table', f'{dataset}.json')
    if os.path.exists(base):
        with open(base, 'r') as f:
            data = json.load(f)
        return data.get('header', None), data.get('rows', [])
    return None, []


def print_paper_style_table(dataset, fixed_header, fixed_rows, momke_row):
    header_labels = fixed_header or ["{a}", "{t}", "{v}", "{a, v}", "{a, t}", "{t, v}", "Average", "{a, t, v}"]
    cond_order = ['a', 't', 'v', 'av', 'at', 'tv', 'avg', 'atv']
    m1, m2 = ('ACC', 'F1')

    print("\n" + "-" * 140)
    print(f"Final Results — {dataset}")
    print("-" * 140)
    h = f"{'Models':<20}" + "".join([f"{c:^16}" for c in header_labels])
    print(h)
    print("-" * 140)

    def row_to_str(model_name, cells):
        out = f"{model_name:<20}"
        for key in cond_order:
            v = cells.get(key, None)
            if v is None or m1 not in v or m2 not in v:
                out += f"{'-':>16}"
            else:
                out += f"{_fmt_pair(v[m1], v[m2]):>16}"
        return out

    for r in fixed_rows:
        print(row_to_str(r.get('model', ''), r.get('cells', {})))

    print(row_to_str('MoMKE (ours)', momke_row))
    print("-" * 140 + "\n")


# NEW: ablation-table printer (rows = ablation setting, columns = conditions)
def print_ablation_table(dataset, ablate_cells):
    """
    ablate_cells: dict like
      {
        'a':   {'ALL_ON': {...}, 'CIT_OFF': {...}, ...},
        't':   {...},
        'atv': {...},
        ...
      }
    """
    cond_order = ['a', 't', 'v', 'av', 'at', 'tv', 'atv']
    header_labels = ["{a}", "{t}", "{v}", "{a, v}", "{a, t}", "{t, v}", "{a, t, v}"]
    tag_order = ['ALL_ON', 'CIT_OFF', 'MRR_OFF', 'RMC_OFF', 'DAE_OFF']
    m1, m2 = ('acc', 'f1')

    print("\n" + "-" * 140)
    print(f"Ablation Results — {dataset}")
    print("-" * 140)
    h = f"{'Ablation':<20}" + "".join([f"{c:^16}" for c in header_labels])
    print(h)
    print("-" * 140)

    for tag in tag_order:
        row = f"{tag:<20}"
        for cond in cond_order:
            cd = ablate_cells.get(cond, {})
            m = cd.get(tag, None)
            if (m is None) or (m1 not in m) or (m2 not in m):
                row += f"{'-':>16}"
            else:
                row += f"{_fmt_pair(m[m1], m[m2]):>16}"
        print(row)
    print("-" * 140 + "\n")


# ------------------------- helpers for paths -------------------------
def features_key(audio_feature, text_feature, video_feature):
    """Compose feature triple key (used for pred_dumps + final table JSON)."""
    return f"{audio_feature}__{text_feature}__{video_feature}"


def results_json_path(args, audio_feature, text_feature, video_feature):
    """Where to store the final compiled table results (JSON)."""
    root = os.path.join(config.LOG_DIR, 'main_result', 'table_cache', args.dataset)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"{features_key(audio_feature, text_feature, video_feature)}.json")


# NEW: ablation JSON path (separate cache)
def ablation_json_path(args, audio_feature, text_feature, video_feature):
    """Where to store the compiled ablation results (JSON)."""
    root = os.path.join(config.LOG_DIR, 'main_result', 'ablation_cache', args.dataset)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"{features_key(audio_feature, text_feature, video_feature)}.json")


# ------------------------- device helper -------------------------
def pick_device(args):
    if args.no_cuda or not torch.cuda.is_available():
        return torch.device('cpu'), False
    n = torch.cuda.device_count()
    if args.gpu < 0 or args.gpu >= n:
        print(f"[device] Requested --gpu {args.gpu}, but {n} visible device(s). Using 0.")
        idx = 0
    else:
        idx = args.gpu
    return torch.device(f'cuda:{idx}'), True


# ------------------------- main -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input features / dataset
    parser.add_argument('--audio-feature', type=str, default=None)
    parser.add_argument('--text-feature', type=str, default=None)
    parser.add_argument('--video-feature', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='CMUMOSEI',
                        choices=['CMUMOSI', 'CMUMOSEI'])

    # Stage-2 contribution switches (also read by model)
    parser.add_argument("--use_cit", type=int, default=1)
    parser.add_argument("--use_mrr", type=int, default=1)
    parser.add_argument("--use_rmc", type=int, default=1)
    parser.add_argument("--use_cmdae", type=int, default=1)

    # Model
    parser.add_argument('--time-attn', '--time_attn', action='store_true', dest='time_attn', default=False)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--drop_rate', type=float, default=0.32)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--n_speakers', type=int, default=1)

    # Training
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num-folder', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_condition', type=str, default='atv',
                        choices=['a', 't', 'v', 'at', 'av', 'tv', 'atv', 'all'])
    parser.add_argument('--stage_epoch', type=float, default=15)

    # Router++ (kept for compatibility, unused by new trainer but some args may be used by model)
    parser.add_argument('--tau_init', type=float, default=5.0)
    parser.add_argument('--tau_final', type=float, default=2.0)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--lb_coef', type=float, default=0.01)
    parser.add_argument('--ent_coef', type=float, default=0.0015)

    # CECAL / UCR-style (kept for compatibility with model, if it uses them internally)
    parser.add_argument('--contrast_tau', type=float, default=0.07)
    parser.add_argument('--lambda_c', type=float, default=0.0)
    parser.add_argument('--lambda_a', type=float, default=0.0)

    # MEM-style reconstruction (reused as CM-DAE weight)
    parser.add_argument('--lambda_m', type=float, default=0.0)

    # Uncertainty-aware routing
    parser.add_argument('--uncert_scale', type=float, default=0.0)

    # Selection & fusion
    parser.add_argument('--select_by', type=str, default='acc', choices=['acc', 'f1'])
    parser.add_argument('--force_uniform_router', action='store_true', default=False)
    parser.add_argument('--no_stage_copy', action='store_true', default=False)

    # Optimization niceties
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # New flags
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.015, help='used if optimizer=adamw')
    parser.add_argument('--cosine_stage2', action='store_true', help='use cosine LR during Stage-2')
    parser.add_argument('--use_huber', action='store_true', help='use Huber loss for regression in Stage-1/2')

    # Dev-based thresholding + prediction dumps
    parser.add_argument('--save_dev_split', type=float, default=0.15,
                        help='fraction of test set to use as dev for thresholding (MOSI only recommended)')
    parser.add_argument('--select_threshold_on', type=str, default='dev', choices=['dev', 'train', 'test'])
    parser.add_argument('--save_preds', action='store_true',
                        help='dump per-sample predictions for best epoch (MOSI/MOSEI)')

    # Optional CCC for MOSI/MOSEI Stage-2
    parser.add_argument('--lambda_ccc', type=float, default=0.0,
                        help='add masked CCC loss to Stage-2 on MOSI/MOSEI')

    # Evaluation pooling
    parser.add_argument('--eval_pool', type=str, default='utter', choices=['utter', 'token'],
                        help='aggregate predictions per utterance before metrics (default: utter)')

    # Ablation flag: test-time switchboard ablations on best checkpoint
    parser.add_argument(
        '--do_switch_ablation',
        action='store_true',
        help='After training, run test-time ablations of CIT/MRR/RMC/CM-DAE on the same best checkpoint.'
    )

    # Resume/Test-only
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--test_only', action='store_true', default=False)

    args = parser.parse_args()
    args.stage_epoch = int(args.stage_epoch)

    device, _is_cuda = pick_device(args)
    args.device = device

    save_folder_name = f'{args.dataset}'
    save_log = os.path.join(config.LOG_DIR, 'main_result', f'{save_folder_name}')
    os.makedirs(save_log, exist_ok=True)

    time_dataset = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_{args.dataset}"
    sys.stdout = Logger(
        filename=f"{save_log}/{time_dataset}_batchsize-{args.batch_size}_lr-{args.lr}_seed-{args.seed}_test-condition-{args.test_condition}.txt",
        stream=sys.stdout
    )

    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch(args.seed)

    # Dataset-specific setup (MOSI/MOSEI only)
    if args.dataset in ['CMUMOSEI', 'CMUMOSI']:
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}. This script is MOSI/MOSEI-only now.")

    print(args)

    print('====== Reading Data =======')
    audio_feature, text_feature, video_feature = args.audio_feature, args.text_feature, args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    print(audio_root)
    print(text_root)
    print(video_root)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(
        video_root), 'features not exist!'

    train_loaders, test_loaders, adim, tdim, vdim = get_loaders(
        audio_root=audio_root,
        text_root=text_root,
        video_root=video_root,
        num_folder=args.num_folder,
        batch_size=args.batch_size,
        dataset=args.dataset,
        num_workers=0
    )
    assert len(train_loaders) == args.num_folder, 'Error: folder number'

    _alignment_probe(args, train_loaders[0], "train")
    _alignment_probe(args, test_loaders[0], "test")

    outdir_preds = None
    if args.save_preds and (args.dataset in ['CMUMOSEI', 'CMUMOSI']):
        outdir_preds = os.path.join(
            config.LOG_DIR, 'main_result', 'pred_dumps', args.dataset,
            features_key(audio_feature, text_feature, video_feature)
        )

    # Run all conditions (a,t,v,av,at,tv,atv) or a single condition.
    if args.test_condition == 'all':
        conds = ['a', 't', 'v', 'av', 'at', 'tv', 'atv']
        momke_cells = {}
        ablate_cells = {}  # NEW: accumulated ablation results across conditions

        for c in conds:
            print('\n' + '=' * 100)
            print(f'Running testing condition {{{c}}}')
            print('=' * 100)
            args.test_condition = c
            res, best_state, ablations = run_one_condition(args, train_loaders, test_loaders, adim, tdim, vdim, outdir_preds)
            momke_cells[c] = res
            if ablations:
                ablate_cells[c] = ablations

            if not args.test_only:
                save_folder_name = f'{args.dataset}'
                save_model = os.path.join(config.MODEL_DIR, 'main_result', save_folder_name)
                os.makedirs(save_model, exist_ok=True)

                suffix_name = f"{time_dataset}_cond-{c}_hidden-{args.hidden}_bs-{args.batch_size}"
                feature_name = f'{audio_feature};{text_feature};{video_feature}'
                res_name = (
                    f"mae-{res.get('MAE', 0):.3f}_corr-{res.get('CORR', 0):.3f}_"
                    f"f1-{res.get('F1', 0):.4f}_acc-{res.get('ACC', 0):.4f}"
                )
                save_path = os.path.join(
                    save_model,
                    f"{suffix_name}_features-{feature_name}_{res_name}_test-condition-{c}.pth"
                )
                torch.save({'result_summary': res,
                            'state_dict': best_state or {}}, save_path)
                print(f"[checkpoint] Saved {c} model to: {save_path}")

            base_keys = [k for k in ['a', 't', 'v', 'av', 'at', 'tv'] if k in momke_cells]
            if base_keys:
                acc = float(np.mean([momke_cells[k]['ACC'] for k in base_keys]))
                f1 = float(np.mean([momke_cells[k]['F1'] for k in base_keys]))
                momke_cells['avg'] = {'ACC': acc, 'F1': f1}

            fixed_header, fixed_rows = load_paper_rows(args.dataset)
            print_paper_style_table(args.dataset, fixed_header, fixed_rows, momke_row=momke_cells)

        base_keys = [k for k in ['a', 't', 'v', 'av', 'at', 'tv'] if k in momke_cells]
        if base_keys:
            acc = float(np.mean([momke_cells[k]['ACC'] for k in base_keys]))
            f1 = float(np.mean([momke_cells[k]['F1'] for k in base_keys]))
            momke_cells['avg'] = {'ACC': acc, 'F1': f1}

        json_out = results_json_path(args, audio_feature, text_feature, video_feature)
        with open(json_out, 'w') as f:
            json.dump(momke_cells, f)
        print(f"[table_cache] Wrote final compiled results to {json_out}")

        # NEW: save and print ablation table
        ablation_out = ablation_json_path(args, audio_feature, text_feature, video_feature)
        with open(ablation_out, 'w') as f:
            json.dump(ablate_cells, f)
        print(f"[ablation_cache] Wrote final ablation results to {ablation_out}")
        if ablate_cells:
            print_ablation_table(args.dataset, ablate_cells)

        sys.exit(0)

    # single-condition run
    res, best_state, ablations = run_one_condition(args, train_loaders, test_loaders, adim, tdim, vdim, outdir_preds)

    # UPDATED: accumulate results across modality runs using the JSON cache
    json_out = results_json_path(args, audio_feature, text_feature, video_feature)
    if os.path.exists(json_out):
        try:
            with open(json_out, 'r') as f:
                momke_cells = json.load(f)
        except Exception:
            momke_cells = {}
    else:
        momke_cells = {}

    # update current condition result
    momke_cells[args.test_condition] = res

    # recompute 'avg' over any base modalities we have so far
    base_keys = [k for k in ['a', 't', 'v', 'av', 'at', 'tv'] if k in momke_cells]
    if base_keys:
        acc = float(np.mean([momke_cells[k]['ACC'] for k in base_keys]))
        f1 = float(np.mean([momke_cells[k]['F1'] for k in base_keys]))
        momke_cells['avg'] = {'ACC': acc, 'F1': f1}

    # save back to JSON cache
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, 'w') as f:
        json.dump(momke_cells, f)
    print(f"[table_cache] Updated results written to {json_out}")

    # NEW: accumulate ablation results across runs
    ablation_out = ablation_json_path(args, audio_feature, text_feature, video_feature)
    if os.path.exists(ablation_out):
        try:
            with open(ablation_out, 'r') as f:
                ablate_cells = json.load(f)
        except Exception:
            ablate_cells = {}
    else:
        ablate_cells = {}

    if ablations:
        ablate_cells[args.test_condition] = ablations

    os.makedirs(os.path.dirname(ablation_out), exist_ok=True)
    with open(ablation_out, 'w') as f:
        json.dump(ablate_cells, f)
    print(f"[ablation_cache] Updated ablations written to {ablation_out}")

    fixed_header, fixed_rows = load_paper_rows(args.dataset)
    print_paper_style_table(args.dataset, fixed_header, fixed_rows, momke_row=momke_cells)

    # NEW: print ablation table at the end of this run
    if ablate_cells:
        print_ablation_table(args.dataset, ablate_cells)

    if args.test_only:
        sys.exit(0)

    print('====== Saving =======')
    save_model = os.path.join(config.MODEL_DIR, 'main_result', f'{save_folder_name}')
    os.makedirs(save_model, exist_ok=True)

    suffix_name = f"{time_dataset}_hidden-{args.hidden}_bs-{args.batch_size}"
    feature_name = f'{audio_feature};{text_feature};{video_feature}'

    res_name = (
        f"mae-{res.get('MAE', 0):.3f}_corr-{res.get('CORR', 0):.3f}_"
        f"f1-{res.get('F1', 0):.4f}_acc-{res.get('ACC', 0):.4f}"
    )

    save_path = (
        f'{save_model}/{suffix_name}_features-{feature_name}_'
        f'{res_name}_test-condition-{args.test_condition}.pth'
    )
    to_save = best_state if best_state is not None else {}
    torch.save({'result_summary': res, 'state_dict': to_save}, save_path)
    print(save_path)
