# utils.py
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# --- make project root importable so we can do `import config` from parent dir ---
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

try:
    from .dataloader_cmumosi import CMUMOSIDataset
    from .model import Rome
except ImportError:
    from dataloader_cmumosi import CMUMOSIDataset
    from model import Rome
import config


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_loaders(audio_root, text_root, video_root, num_folder, dataset, batch_size, num_workers):
    ## CMU datasets
    if dataset in ['CMUMOSI', 'CMUMOSEI']:
        dataset = CMUMOSIDataset(label_path=config.PATH_TO_LABEL[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)
        trainNum = len(dataset.trainVids)
        valNum = len(dataset.valVids)
        testNum = len(dataset.testVids)
        train_idxs = list(range(0, trainNum))
        val_idxs = list(range(trainNum, trainNum+valNum))
        test_idxs = list(range(trainNum+valNum, trainNum+valNum+testNum))

        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  sampler=SubsetRandomSampler(train_idxs),
                                  collate_fn=dataset.collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=False)
        test_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 sampler=SubsetRandomSampler(test_idxs),
                                 collate_fn=dataset.collate_fn,
                                 num_workers=num_workers,
                                 pin_memory=False)
        train_loaders = [train_loader]
        test_loaders = [test_loader]

        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, test_loaders, adim, tdim, vdim

    raise ValueError(
        f"Unsupported dataset {dataset}. Rome supports only CMUMOSI and CMUMOSEI."
    )


def build_model(args, adim, tdim, vdim):
    D_e = args.hidden
    model = Rome(args,
                  adim, tdim, vdim, D_e,
                  n_classes=args.n_classes,
                  depth=args.depth, num_heads=args.num_heads, mlp_ratio=1, drop_rate=args.drop_rate,
                  attn_drop_rate=args.attn_drop_rate,
                  no_cuda=args.no_cuda)
    print("Model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    return model


def generate_mask(seqlen, batch, test_condition, first_stage):
    """Simulate partial-view data with complete view data by masking modalities."""
    if first_stage:
        audio_mask = np.array([1])
        text_mask = np.array([1])
        visual_mask = np.array([1])
    else:
        audio_mask = np.array([1 if 'a' in test_condition else 0])
        text_mask  = np.array([1 if 't' in test_condition else 0])
        visual_mask= np.array([1 if 'v' in test_condition else 0])
    audio_mask = audio_mask.repeat(seqlen * batch)
    text_mask  = text_mask.repeat(seqlen * batch)
    visual_mask= visual_mask.repeat(seqlen * batch)
    return [audio_mask, text_mask, visual_mask]


## gain input features: ?*[seqlen, batch, dim]
def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    input_features = []
    feat1 = torch.cat([audio_host, text_host, visual_host], dim=2) # [seqlen, batch, adim+tdim+vdim]
    feat2 = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    featdim = feat1.size(-1)
    tmask = qmask.transpose(0, 1)                 # [batch, seqlen] -> [seqlen, batch]
    tmask = tmask.unsqueeze(2).repeat(1, 1, featdim) # -> [seqlen, batch, featdim]
    select_feat = torch.where(tmask == 0, feat1, feat2) # -> [seqlen, batch, featdim]
    input_features.append(select_feat)
    return input_features
