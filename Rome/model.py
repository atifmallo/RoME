import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .modules.Attention_softmoe import *
except ImportError:
    from modules.Attention_softmoe import *


_EPS = 1e-8


class Rome(nn.Module):
    """
    Rome with 4 Stage-2 contributions:

      1) CIT   - Cross-Modal Interaction Transformer over joint [A,T,V] features.
      2) MRR   - Modality Reliability & Re-weighting for late fusion.
      3) RMC   - Residual Multimodal Corrector on top of the fused baseline.
      4) CM-DAE - Cross-Modal Denoising Autoencoder over the CIT latent.

    """

    def __init__(self, args, adim, tdim, vdim, D_e, n_classes,
                 depth=4, num_heads=4, mlp_ratio=1,
                 drop_rate=0.0, attn_drop_rate=0.0, no_cuda=False):
        super(Rome, self).__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        self.num_heads = num_heads
        D = 3 * D_e
        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate

        # ---------------- Switchboard flags (Stage-2) ----------------
        
        self.use_cit = bool(getattr(args, "use_cit", True))
        self.use_mrr = bool(getattr(args, "use_mrr", True))
        self.use_rmc = bool(getattr(args, "use_rmc", True))
        self.use_cmdae = bool(getattr(args, "use_cmdae", True))

        # -------- input projections (unchanged) --------
        self.a_in_proj = nn.Sequential(nn.Linear(self.adim, D_e))
        self.t_in_proj = nn.Sequential(nn.Linear(self.tdim, D_e))
        self.v_in_proj = nn.Sequential(nn.Linear(self.vdim, D_e))
        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)

        # temporal / expert encoder block (shared Stage-1/Stage-2)
        self.block = Block(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )

        # Stage-1 fusion head (unchanged)
        D_joint = D  # 3 * D_e
        self.D_joint = D_joint
        self.proj1 = nn.Linear(D_joint, D_joint)
        self.nlp_head_a = nn.Linear(D_e, n_classes)
        self.nlp_head_t = nn.Linear(D_e, n_classes)
        self.nlp_head_v = nn.Linear(D_e, n_classes)
        self.nlp_head = nn.Linear(D_joint, n_classes)

        # Routers (for 3 experts per modality)
        self.router_a = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )
        self.router_t = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )
        self.router_v = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )

        # ================= STAGE-2 CONTRIBUTIONS =================

        # 1) CIT – Cross-Modal Interaction Transformer over joint [A,T,V] features.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_joint,
            nhead=num_heads,
            dim_feedforward=2 * D_joint,
            dropout=drop_rate,
            batch_first=True,
        )
        self.cit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 2) MRR – Modality Reliability & Re-weighting.
        rel_hidden = max(D_e, 32)
        self.rel_a = nn.Sequential(
            nn.Linear(D_e + 1, rel_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(rel_hidden, 1),
            nn.Sigmoid(),
        )
        self.rel_t = nn.Sequential(
            nn.Linear(D_e + 1, rel_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(rel_hidden, 1),
            nn.Sigmoid(),
        )
        self.rel_v = nn.Sequential(
            nn.Linear(D_e + 1, rel_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(rel_hidden, 1),
            nn.Sigmoid(),
        )

        # 3) RMC – Residual Multimodal Corrector.
        #    IMPORTANT: this will now see F_latent (DAE-shaped) instead of raw F_global.
        self.rmc_head = nn.Sequential(
            nn.Linear(D_joint, D_joint),
            nn.ReLU(inplace=True),
            nn.Dropout(self.out_dropout),
            nn.Linear(D_joint, n_classes),
        )

        # 4) CM-DAE – Cross-Modal Denoising Autoencoder decoders.
        #    They reconstruct pooled unimodal features z_a/z_t/z_v from
        #    a bottleneck latent F_latent derived from F_global.
        self.cmdae_bottleneck = nn.Sequential(
            nn.Linear(D_joint, D_joint),
            nn.ReLU(inplace=True),
            nn.Dropout(self.out_dropout),
        )

        self.cmdae_dec_a = nn.Sequential(
            nn.Linear(D_joint, D_joint),
            nn.ReLU(inplace=True),
            nn.Dropout(self.out_dropout),
            nn.Linear(D_joint, D_e),
        )
        self.cmdae_dec_t = nn.Sequential(
            nn.Linear(D_joint, D_joint),
            nn.ReLU(inplace=True),
            nn.Dropout(self.out_dropout),
            nn.Linear(D_joint, D_e),
        )
        self.cmdae_dec_v = nn.Sequential(
            nn.Linear(D_joint, D_joint),
            nn.ReLU(inplace=True),
            nn.Dropout(self.out_dropout),
            nn.Linear(D_joint, D_e),
        )

        # cache for CM-DAE (for external loss computation in training code)
        self.cmdae_cache = None

    def cuda(self, device=None):
        super(Rome, self).cuda(device)
        return self

    # ----------------------------------------------------------------- #
    #                              FORWARD                              #
    # ----------------------------------------------------------------- #
    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False):
        """
        inputfeats:          [seqlen, batch, adim+tdim+vdim]
        input_features_mask: [seqlen, batch, 3]  (a/t/v availability)
        umask:               [batch, seqlen] (valid utterance positions)
        first_stage:         True  => Stage-1 unimodal + simple fusion (unchanged)
                             False => Stage-2 with CIT+MRR+RMC(+CM-DAE)
        """
        weight_save = []

        # clear CM-DAE cache at each call
        self.cmdae_cache = None

        # ----- split raw features -----
        audio = inputfeats[:, :, :self.adim]
        text = inputfeats[:, :, self.adim:self.adim + self.tdim]
        video = inputfeats[:, :, self.adim + self.tdim:]

        seq_len, B, _ = audio.shape

        # --> [batch, seqlen, dim]
        audio = audio.permute(1, 0, 2)
        text = text.permute(1, 0, 2)
        video = video.permute(1, 0, 2)

        proj_a = self.dropout_a(self.a_in_proj(audio))
        proj_t = self.dropout_t(self.t_in_proj(text))
        proj_v = self.dropout_v(self.v_in_proj(video))

        # --> [batch, seqlen, 3]
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))  # [B,L,3]
        if umask is not None:
            input_mask[umask == 0] = 0
        # --> [batch, 3, seqlen] -> [batch, 3*seqlen]
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)  # [B,3L]

        # ---------- routers (weights over 3 experts per modality) ----------
        weight_a = self.router_a(proj_a)  # [B,L,3]
        weight_t = self.router_t(proj_t)
        weight_v = self.router_v(proj_v)

        weight_a = torch.softmax(weight_a, dim=-1)
        weight_t = torch.softmax(weight_t, dim=-1)
        weight_v = torch.softmax(weight_v, dim=-1)

        weight_save.append(
            np.array([
                weight_a.detach().cpu().numpy(),
                weight_t.detach().cpu().numpy(),
                weight_v.detach().cpu().numpy()
            ])
        )

        weight_a_exp = weight_a.unsqueeze(-1).repeat(1, 1, 1, self.D_e)  # [B,L,3,D_e]
        weight_t_exp = weight_t.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        weight_v_exp = weight_v.unsqueeze(-1).repeat(1, 1, 1, self.D_e)

        # ---------- temporal / expert encoders ----------
        x_a = self.block(proj_a, first_stage, attn_mask, 'a')  # [B,3L,D_e]
        x_t = self.block(proj_t, first_stage, attn_mask, 't')  # [B,3L,D_e]
        x_v = self.block(proj_v, first_stage, attn_mask, 'v')  # [B,3L,D_e]

        # ========================= STAGE-1 =========================
        if first_stage:
            # Unimodal heads as in original file
            out_a = self.nlp_head_a(x_a)  # [B,3L,1]
            out_t = self.nlp_head_t(x_t)
            out_v = self.nlp_head_v(x_v)

            # Simple fusion over concatenated sequence (unchanged)
            x = torch.cat([x_a, x_t, x_v], dim=1)  # [B,9L,D_e]
            x[attn_mask == 0] = 0  # broadcast over last dim

            # split back per modality to length L and fuse in feature dimension
            x_a_s = x[:, :seq_len, :]
            x_t_s = x[:, seq_len:2 * seq_len, :]
            x_v_s = x[:, 2 * seq_len:, :]
            x_joint = torch.cat([x_a_s, x_t_s, x_v_s], dim=-1)  # [B,L,3D_e]

            res = x_joint
            u = F.relu(self.proj1(x_joint))
            u = F.dropout(u, p=self.out_dropout, training=self.training)
            hidden = u + res
            out = self.nlp_head(hidden)  # [B,L,1]

            return hidden, out, out_a, out_t, out_v, np.array(weight_save)

        # ========================= STAGE-2 =========================
        # NOTE: Encoders are the same; we only add contributions on top.

        # ---------- Switchboard (Stage-2) ----------
        use_cit = bool(self.use_cit)
        use_mrr = bool(self.use_mrr)
        use_rmc = bool(self.use_rmc)
        use_cmdae = bool(self.use_cmdae)

        if not hasattr(self, "_switchboard_printed"):
            print(
                f"[Rome/SW-4C] Stage-2 contributions: "
                f"CIT={use_cit}  MRR={use_mrr}  RMC={use_rmc}  CM-DAE={use_cmdae}",
            )
            self._switchboard_printed = True

        # --- expert mixing (same as 3-contrib Stage-2) ---
        x_unweighted_a = x_a.reshape(B, seq_len, 3, self.D_e)   # [B,L,3,D_e]
        x_unweighted_t = x_t.reshape(B, seq_len, 3, self.D_e)
        x_unweighted_v = x_v.reshape(B, seq_len, 3, self.D_e)

        x_out_a = torch.sum(weight_a_exp * x_unweighted_a, dim=2)  # [B,L,D_e]
        x_out_t = torch.sum(weight_t_exp * x_unweighted_t, dim=2)
        x_out_v = torch.sum(weight_v_exp * x_unweighted_v, dim=2)

        # Unimodal predictions from routed features
        out_a = self.nlp_head_a(x_out_a)  # [B,L,1]
        out_t = self.nlp_head_t(x_out_t)
        out_v = self.nlp_head_v(x_out_v)

        # Concatenate routed features along sequence dimension
        x = torch.cat([x_out_a, x_out_t, x_out_v], dim=1)  # [B,3L,D_e]
        x[attn_mask == 0] = 0

        # Split back by modality and build joint per-time representation
        x_a_s = x[:, :seq_len, :]               # [B,L,D_e]
        x_t_s = x[:, seq_len:2 * seq_len, :]    # [B,L,D_e]
        x_v_s = x[:, 2 * seq_len:, :]           # [B,L,D_e]
        x_joint = torch.cat([x_a_s, x_t_s, x_v_s], dim=-1)  # [B,L,3D_e] = [B,L,D_joint]

        # ---------- MRR: modality reliability weights ----------
        if umask is None:
            umask = torch.ones(B, seq_len, device=x_joint.device, dtype=torch.long)
        m = umask.float()  # [B,L]
        msum = m.sum(dim=1, keepdim=True).clamp_min(_EPS)

        if use_mrr:
            # pooled routed features per modality (for reliability)
            z_a_mrr = (x_out_a * m.unsqueeze(-1)).sum(dim=1) / msum  # [B,D_e]
            z_t_mrr = (x_out_t * m.unsqueeze(-1)).sum(dim=1) / msum
            z_v_mrr = (x_out_v * m.unsqueeze(-1)).sum(dim=1) / msum

            # pooled unimodal logits
            ya = (out_a.squeeze(-1) * m).sum(dim=1, keepdim=True) / msum  # [B,1]
            yt = (out_t.squeeze(-1) * m).sum(dim=1, keepdim=True) / msum
            yv = (out_v.squeeze(-1) * m).sum(dim=1, keepdim=True) / msum

            # reliability scores in [0,1]
            ra = self.rel_a(torch.cat([z_a_mrr, ya], dim=-1))  # [B,1]
            rt = self.rel_t(torch.cat([z_t_mrr, yt], dim=-1))
            rv = self.rel_v(torch.cat([z_v_mrr, yv], dim=-1))

            # broadcast over time & channels
            ra_b = ra.unsqueeze(1)  # [B,1,1]
            rt_b = rt.unsqueeze(1)
            rv_b = rv.unsqueeze(1)

            r_sum = (ra_b + rt_b + rv_b).clamp_min(_EPS)
            # baseline reliability-weighted fusion (per token)
            y_base = (ra_b * out_a + rt_b * out_t + rv_b * out_v) / r_sum  # [B,L,1]
        else:
            # simple uniform average baseline when MRR is off
            y_base = (out_a + out_t + out_v) / 3.0  # [B,L,1]

        # ---------- CIT: cross-modal interaction over joint [A,T,V] ----------
        cit_mask = (umask == 0)  # [B,L], bool mask for padded timesteps
        if use_cit:
            x_cit = self.cit_encoder(x_joint, src_key_padding_mask=cit_mask)  # [B,L,3D_e]
        else:
            # no cross-modal interaction when CIT is off
            x_cit = x_joint

        # Global multimodal representation (pooled over valid timesteps)
        F_global = (x_cit * m.unsqueeze(-1)).sum(dim=1) / msum  # [B,3D_e]

        # ---------- CM-DAE: cross-modal denoising autoencoder ----------
        # IMPORTANT: F_latent is what RMC will see. DAE ON/OFF changes F_latent.
        if use_cmdae:
            # DAE-shaped latent
            F_latent = self.cmdae_bottleneck(F_global)  # [B,3D_e]

            # pooled original unimodal features (for reconstruction targets)
            z_a = (x_out_a * m.unsqueeze(-1)).sum(dim=1) / msum  # [B,D_e]
            z_t = (x_out_t * m.unsqueeze(-1)).sum(dim=1) / msum
            z_v = (x_out_v * m.unsqueeze(-1)).sum(dim=1) / msum

            # reconstructions from shared latent
            z_hat_a = self.cmdae_dec_a(F_latent)  # [B,D_e]
            z_hat_t = self.cmdae_dec_t(F_latent)
            z_hat_v = self.cmdae_dec_v(F_latent)

            # cache for external reconstruction loss in training code
            self.cmdae_cache = {
                "z_a": z_a,
                "z_t": z_t,
                "z_v": z_v,
                "z_hat_a": z_hat_a,
                "z_hat_t": z_hat_t,
                "z_hat_v": z_hat_v,
            }
        else:
            # no DAE shaping: latent is just raw F_global
            F_latent = F_global
            self.cmdae_cache = None

        # ---------- RMC: residual multimodal corrector ----------
        if use_rmc:
            # NOTE: now uses F_latent instead of F_global
            delta = self.rmc_head(F_latent)  # [B,1]
            delta = delta.unsqueeze(1)       # [B,1,1], broadcast over time
        else:
            # no correction when RMC is off
            delta = torch.zeros_like(y_base[:, :1, :])

        out = y_base + delta  # [B,L,1]
        hidden = x_cit        # expose CIT features as "hidden" for downstream analysis

        return hidden, out, out_a, out_t, out_v, np.array(weight_save)


if __name__ == '__main__':
    # simple shape sanity-check (not a full test)
    B, L = 4, 10
    adim = tdim = vdim = 8
    D_e = 12
    dummy_args = type('obj', (), {})()
    dummy_args.device = torch.device('cpu')
    dummy_args.drop_rate = 0.1
    # switches default to True if not present

    model = Rome(dummy_args, adim, tdim, vdim, D_e, n_classes=1)
    feats = torch.randn(L, B, adim + tdim + vdim)
    mask_feats = torch.ones(L, B, 3)
    umask = torch.ones(B, L, dtype=torch.long)

    # Stage-1
    h1, o1, oa1, ot1, ov1, w1 = model(feats, mask_feats, umask, first_stage=True)
    # Stage-2
    h2, o2, oa2, ot2, ov2, w2 = model(feats, mask_feats, umask, first_stage=False)
    print("Stage-1 out:", o1.shape, "Stage-2 out:", o2.shape)
