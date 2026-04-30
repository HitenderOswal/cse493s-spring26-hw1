import argparse
import json
import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import GPTConfig, GPT

parser = argparse.ArgumentParser(description='Train model from config.')
parser.add_argument('--train-config', type=str, required=True, help='Path to training config.')
parser.add_argument('--model-config', type=str, required=True, help='Path to model config.')
parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint.")
args = parser.parse_args()

with open(args.train_config, 'r', encoding='utf-8') as file:
    train_config = json.load(file)

with open(args.model_config, 'r', encoding='utf-8') as file:
    model_config = json.load(file)

# if not torch.cuda.is_available():
#     raise RuntimeError('No GPU available.')

train_config['dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

os.makedirs(train_config['experiment_dir'], exist_ok=True)
tb_log_dir = os.path.join(train_config['experiment_dir'], 'tensorboard')
writer = SummaryWriter(log_dir=tb_log_dir)

torch.manual_seed(train_config['seed'])
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = np.load(os.path.join(train_config['dataset'], 'train.npy'))
val_data = np.load(os.path.join(train_config['dataset'], 'val.npy'))
test_data = np.load(os.path.join(train_config['dataset'], 'test.npy'))

# def get_batch(split):
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - model_config['block_size'], (train_config['batch_size'],)).tolist()
#     x = torch.stack([torch.from_numpy((data[i:i+model_config['block_size']]).astype(np.int64)) for i in ix])
#     y = torch.stack([torch.from_numpy((data[i+1:i+1+model_config['block_size']]).astype(np.int64)) for i in ix])
#     x, y = x.pin_memory().to(train_device, non_blocking=True), y.pin_memory().to(train_device, non_blocking=True)
#     return x, y

def get_batch(split):
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data), (train_config['batch_size'],))
    batch = torch.from_numpy(data[ix]).long()

    x = batch[:, :-1]
    y = batch[:, -1]

    x = x.to(train_device)
    y = y.to(train_device)

    return x, y

iter_num = 0
best_val_loss = 1e9

model = GPT(GPTConfig(**model_config))

if args.resume and os.path.exists(os.path.join(train_config['experiment_dir'], 'ckpt.pt')):
    print(f"Resuming training from {train_config['experiment_dir']}")
    ckpt_path = os.path.join(train_config['experiment_dir'], 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=train_device)
    model.load_state_dict(checkpoint['model'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(train_device)

scaler = torch.amp.GradScaler(enabled=(train_config['dtype'] == torch.float16))

optimizer = model.configure_optimizers(train_config['weight_decay'], train_config['learning_rate'], (train_config['beta1'], train_config['beta2']), device_type)

if args.resume:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None
raw_model = model
if train_config['compile']:
    model = torch.compile(model)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(train_config['eval_iters'], device=train_device)
        accs = torch.zeros(train_config['eval_iters'], device=train_device)

        for k in range(train_config['eval_iters']):
            X, Y = get_batch(split)
            with torch.amp.autocast(device_type=device_type, dtype=train_config['dtype']):
            #     logits = model(X)
            #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))

                logits = model(X)
                logits = logits[:, -1, :]
                loss = F.cross_entropy(logits, Y)

            preds = torch.argmax(logits, dim=-1)
            acc = (preds == Y).float().mean()

            losses[k] = loss.item()
            accs[k] = acc.item()

        out[split] = losses.mean()
        out[split + '_acc'] = accs.mean().item()

    model.train()
    return out

def get_lr(it):
    if it < train_config['warmup_iters']:
        return train_config['learning_rate'] * (it + 1) / (train_config['warmup_iters'] + 1)
    if it > train_config['lr_decay_iters']:
        return train_config['min_lr']
    decay_ratio = (it - train_config['warmup_iters']) / (train_config['lr_decay_iters'] - train_config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config['min_lr'] + coeff * (train_config['learning_rate'] - train_config['min_lr'])

X, Y = get_batch('train')

while True:
    lr = get_lr(iter_num) if train_config['decay_lr'] else train_config['learning_rate']
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % train_config['eval_interval'] == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train acc {losses['train_acc']:.4f}, val acc {losses['val_acc']:.4f}")
        writer.add_scalar('loss/train_eval', losses['train'].item(), iter_num)
        writer.add_scalar('loss/val', losses['val'].item(), iter_num)
        writer.add_scalar('lr', lr, iter_num)
        writer.add_scalar('acc/train_eval', losses['train_acc'], iter_num)
        writer.add_scalar('acc/val', losses['val_acc'], iter_num)
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'train_config': train_config,
            'model_config': model_config,
        }
        print(f"saving checkpoint to {train_config['experiment_dir']}")
        torch.save(checkpoint, os.path.join(train_config['experiment_dir'], 'ckpt.pt'))

    for _ in range(train_config['gradient_accumulation_steps']):
        with torch.amp.autocast(device_type=device_type, dtype=train_config['dtype']):
            logits = model(X)
            logits = logits[:, -1, :]
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            loss = F.cross_entropy(logits, Y)
            loss = loss / train_config['gradient_accumulation_steps']
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if train_config['grad_clip'] != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip'])

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if iter_num % train_config['log_interval'] == 0:
        writer.add_scalar(
            'loss/train_step',
            loss.item() * train_config['gradient_accumulation_steps'],
            iter_num,
        )

    iter_num += 1

    if iter_num > train_config['max_iters']:
        break

writer.close()