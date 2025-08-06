import math
import copy
from pathlib import Path
import random
from collections import namedtuple, Counter
from multiprocessing import cpu_count
import os
import re
from datetime import datetime
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from contextlib import nullcontext
import json
import pandas as pd

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from tqdm.auto import tqdm

from transformers import get_scheduler
from transformers import (
    BertTokenizerFast,
    BartForConditionalGeneration,
    BartConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from accelerate import Accelerator, DistributedDataParallelKwargs
import wandb

from torch.optim.optimizer import Optimizer
from torch.optim import AdamW
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader

from traj_compressed_ae import BARTLatentCompression


os.environ["WANDB_MODE"] = "offline"


generate_kwargs = {
    'beam':
        {'max_length': 64, 'min_length': 5, 'do_sample': False, 'num_beams': 4, 'no_repeat_ngram_size': 3,
         'repetition_penalty': 1.2},
    'nucleus':
        {'max_length': 64, 'min_length': 5, 'do_sample': True, 'top_p': .95, 'num_beams': 1, 'no_repeat_ngram_size': 3,
         'repetition_penalty': 1.2}}

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# helpers functions

def get_output_dir(args):
    model_dir = f'{Path(args.dataset_name).stem}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    output_dir = os.path.join(args.save_dir, model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Created {output_dir}')
    return output_dir

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_sequence(seq):
    if isinstance(seq, str):
        return [int(x) for x in seq.strip().split()]
    elif isinstance(seq, list):
        return seq
    else:
        raise ValueError("unknown type")


def compute_grad_norm(parameters):
    # implementation adapted from https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=2) for p in parameters]), p=2).item()
    return total_norm

# functions

def separate_weight_decayable_params(params):
    # Exclude affine params in norms (e.g. LayerNorm, GroupNorm, etc.) and bias terms
    no_wd_params = [param for param in params if param.ndim < 2]
    wd_params = [param for param in params if param not in set(no_wd_params)]
    return wd_params, no_wd_params

def get_adamw_optimizer(params, lr, betas, weight_decay, eps=1e-8):
    params = list(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(param_groups, lr = lr, weight_decay = weight_decay, betas=betas, eps=eps)

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, tokenizer, segment_coord_map, max_length=128):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.segment_coord_map = segment_coord_map

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_length]
        text = " ".join(str(x) for x in seq)
        lat = [self.segment_coord_map.get(x, (0.0, 0.0))[0] for x in seq]
        lon = [self.segment_coord_map.get(x, (0.0, 0.0))[1] for x in seq]
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        item = {key: encoding[key].squeeze(0) for key in encoding}
        item["lat"] = torch.tensor(lat + [0.0] * (self.max_length - len(lat)))
        item["lon"] = torch.tensor(lon + [0.0] * (self.max_length - len(lon)))
        item["labels"] = item["input_ids"].clone()
        return item


# trainer class

class Trainer(object):
    def __init__(
            self,
            args,
            dataset_name,
            *,
            train_batch_size=16,
            eval_batch_size=64,
            train_lr=1e-4,
            train_num_steps=100000,
            lr_schedule='cosine',
            num_warmup_steps=500,
            adam_betas=(0.9, 0.99),
            adam_weight_decay=0.01,
            num_samples=None,
            eval_every=1000,
            results_folder='./results',
            mixed_precision='no',
            seed=43,
            use_coords=False
    ):
        super().__init__()

        set_seeds(seed)

        self.args = args

        self.best_val_metric = 0
        self.num_samples = num_samples

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            log_with='wandb',
            kwargs_handlers=[ddp_kwargs]
        )
        self.num_devices = self.accelerator.num_processes
        args.num_devices = self.num_devices

        if self.accelerator.is_main_process:
            if args.output_dir is None:
                args.output_dir = get_output_dir(args)
            results_folder = args.output_dir
            with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args,
                                               init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})

        self.enc_dec_model = args.enc_dec_model

        config = BartForConditionalGeneration.from_pretrained(
            args.enc_dec_model).config
        self.lm = BARTLatentCompression.from_pretrained(
            args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents,
            num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers,
            l2_normalize_latents=args.l2_normalize_latents, _fast_init=False,use_coords=use_coords)

        if args.lm_mode == 'ft':
            for (param_name, param) in self.lm.named_parameters():
                param.requires_grad = True
        elif args.lm_mode == 'freeze':
            for (param_name, param) in self.lm.named_parameters():
                if re.fullmatch(".*perceiver.*", param_name):
                    param.requires_grad = True
                    print(f"Trainable: {param_name}")
                else:
                    param.requires_grad = False

        num_trainable_params = sum(p.numel() for p in self.lm.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            self.accelerator.print(f'num trainable params: {num_trainable_params}')

        self.eval_every = eval_every

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.train_num_steps = train_num_steps

        # ==========  dataset and dataloader ==========
        # obtain centroid latitude and longitude of segments
        segment_df = pd.read_csv("/storage/baoshen/public_dataset/final_didi_train_segment_generation/edge_map_feature_chengdu.csv")
        lat_min, lat_max = segment_df["lat"].min(), segment_df["lat"].max()
        lon_min, lon_max = segment_df["lon"].min(), segment_df["lon"].max()
        segment_df["norm_lat"] = 2 * (segment_df["lat"] - lat_min) / (lat_max - lat_min) - 1
        segment_df["norm_lon"] = 2 * (segment_df["lon"] - lon_min) / (lon_max - lon_min) - 1
        segment_coord_map = {
            int(row.edge_id): (row.norm_lat, row.norm_lon) for _, row in segment_df.iterrows()
        }

        file_path = "/storage/baoshen/public_dataset/final_didi_train_segment_generation/train_data/final_segments_all_train_data.pkl"
        trajs = pd.read_pickle(file_path)
        all_sequences = trajs['unique_id_seq'].apply(parse_sequence).tolist()
        max_seq_length = 128

        vocab_file = "/storage/baoshen/cardiff/cardiff_lm/pretrain_lm/bart_vocab.txt"
        self.tokenizer = BertTokenizerFast(vocab_file=vocab_file, do_lower_case=False)
        self.tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
            "unk_token": "<unk>"
        })

        dataset = TrajectoryDataset(all_sequences, self.tokenizer, segment_coord_map, max_length=max_seq_length)
        train_size = int(0.9 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.lm, padding="longest")

        self.dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                      collate_fn=data_collator)
        self.val_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                     collate_fn=data_collator)

        self.max_seq_len = args.max_seq_len

        # optimizer
        if args.optimizer == 'adamw':
            self.opt = get_adamw_optimizer(self.lm.parameters(), lr=train_lr, betas=adam_betas,
                                                     weight_decay=adam_weight_decay)
        else:
            raise ValueError(f'invalid optimizer {args.optimizer}')

        # scheduler
        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps * self.num_devices,
            num_training_steps=train_num_steps * self.num_devices,
        )

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0
        (self.lm, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader) = self.accelerator.prepare(self.lm,
                                                                            self.opt,
                                                                            self.dataloader,
                                                                            lr_scheduler,
                                                                            self.val_dataloader)
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)

    def save(self):
        if not self.accelerator.is_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.lm),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, file_path=None, resume_training=False):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.lm)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        # Advance lr scheduler to current step
        if resume_training:
            for _ in range(self.step):
                self.lr_scheduler.step()

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        self.lm.train()
        if self.args.lm_mode == 'freeze':
            encoder_context = torch.no_grad()
        else:
            encoder_context = nullcontext()

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.
                data = {k: v.to(device) for k, v in next(self.data_iter).items()}

                with accelerator.autocast():
                    with encoder_context:
                        if self.num_devices > 1:
                            encoder_outputs = self.lm.module.get_encoder()(input_ids=data['input_ids'],
                                                                           attention_mask=data['attention_mask'])
                        else:
                            encoder_outputs = self.lm.get_encoder()(input_ids=data['input_ids'],
                                                                    attention_mask=data['attention_mask'])
                    if self.num_devices > 1:
                        encoder_outputs = self.lm.module.encoder_output_to_decoder_input(encoder_outputs,
                                                                                         data['attention_mask'],
                                                                                         segment_coords=torch.stack([data['lat'], data['lon']], dim=-1))
                    else:
                        encoder_outputs = self.lm.encoder_output_to_decoder_input(encoder_outputs,
                                                                                  data['attention_mask'],
                                                                                  segment_coords=torch.stack([data['lat'], data['lon']], dim=-1))

                    loss = self.lm(labels=data['labels'], encoder_outputs=encoder_outputs).loss
                    # latent regularization
                    loss = loss
                total_loss += loss.item()
                self.accelerator.backward(loss)

                accelerator.wait_for_everyone()

                grad_norm = compute_grad_norm(self.lm.parameters())

                accelerator.clip_grad_norm_(self.lm.parameters(), 1.0)
                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                # Log to WandB
                if self.step % 50 == 0:
                    self.lm.eval()
                    with torch.no_grad():
                        total_val_loss = 0.
                        total_lm_val_loss = 0.
                        data = {k: v.to(device) for k, v in next(self.val_iter).items()}

                        if self.num_devices > 1:
                            encoder_outputs = self.lm.module.get_encoder()(input_ids=data['input_ids'],
                                                                           attention_mask=data['attention_mask'])
                            encoder_outputs = self.lm.module.encoder_output_to_decoder_input(encoder_outputs,
                                                                                             data['attention_mask'],
                                                                                             segment_coords=torch.stack([data['lat'], data['lon']], dim=-1))
                        else:
                            encoder_outputs = self.lm.get_encoder()(input_ids=data['input_ids'],
                                                                    attention_mask=data['attention_mask'])
                            encoder_outputs = self.lm.encoder_output_to_decoder_input(encoder_outputs,
                                                                                      data['attention_mask'],
                                                                                      segment_coords=torch.stack([data['lat'], data['lon']], dim=-1))
                        loss = self.lm(labels=data['labels'], encoder_outputs=encoder_outputs).loss
                        if self.args.lm_mode == 'freeze':
                            total_lm_val_loss += self.lm(input_ids=data['input_ids'],
                                                         attention_mask=data['attention_mask'],
                                                         labels=data['labels']).loss.item()
                        total_val_loss += loss.item()

                        logs = {"train/loss": total_loss, "val/loss": total_val_loss, "grad_norm": grad_norm,
                                "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step,
                                "epoch": (self.step) / len(self.dataloader),
                                "samples": self.step * self.train_batch_size * self.num_devices}
                        if self.args.lm_mode == 'freeze':
                            logs["val/lm_loss"] = total_lm_val_loss
                        pbar.set_postfix(**logs)

                    self.lm.train()
                else:
                    logs = {"train/loss": total_loss, "grad_norm": grad_norm, "lr": self.lr_scheduler.get_last_lr()[0],
                            "step": self.step, "epoch": (self.step) / len(self.dataloader),
                            "samples": self.step * self.train_batch_size * self.num_devices}

                if accelerator.is_main_process:
                    accelerator.log(logs, step=self.step)

                if self.step % self.eval_every == 0:
                    accelerator.wait_for_everyone()
                    self.save()
                    self.lm.train()

                pbar.update(1)
        self.save()
        accelerator.print('training complete')