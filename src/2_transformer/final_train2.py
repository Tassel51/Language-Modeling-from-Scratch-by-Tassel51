import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import time
import pickle
import argparse
import numpy as np
import torch
import wandb

from adamw import AdamW
from dataloader import DataLoader
from transformermodule import TransformerModule
from transformermodule_withoutrmsnorm import TransformerModuleWithoutRMSNorm
from cross_entropy import CrossEntropyLoss
from lr_cosine_shedule import CosineSchedule

# 接着运行指令
# python final_train2.py --device cuda:0 --resume_from checkpoints/model_epoch_4_20260410_203000.pth


# =========================
# 参数
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--train_steps", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--valid_batch_size", type=int, default=2)
parser.add_argument("--max_val_batches", type=int, default=100)
parser.add_argument("--resume_from", type=str, default=None)
parser.add_argument(
    "--no-rmsnorm",
    dest="use_rmsnorm",
    action="store_false",
    help="Disable RMSNorm and use LayerNorm instead"
)
parser.set_defaults(use_rmsnorm=True)
args = parser.parse_args()

device_arg = args.device
epochs = args.epochs
train_steps = args.train_steps
batch_size = args.batch_size
valid_batch_size = args.valid_batch_size
max_val_batches = args.max_val_batches
resume_from = args.resume_from

timestamp = time.strftime("%Y%m%d_%H%M%S")


# =========================
# 设备选择
# =========================
if torch.cuda.is_available() and "cuda" in device_arg:
    device = torch.device(device_arg)
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# =========================
# 路径配置
# =========================
train_data_path = "../data/TinyStoriesV2-GPT4-train.txt"
valid_data_path = "../data/TinyStoriesV2-GPT4-valid.txt"

vocab_pkl_path = "pkls/vocab.pkl"
merges_pkl_path = "pkls/merges.pkl"

encoded_train_path = "pkls/encoded_ids_train.pkl"
encoded_valid_path = "pkls/encoded_ids_valid.pkl"

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


# =========================
# 自动读取真实 vocab_size
# =========================
with open(vocab_pkl_path, "rb") as f:
    vocab = pickle.load(f)

real_vocab_size = len(vocab)
print(f"real_vocab_size = {real_vocab_size}")


# =========================
# wandb
# =========================
wandb.login()
run = wandb.init(
    entity="2775297866-neu",
    project="cs336_final_train",
    config={
        "experiment_name": f"tinystories_17M_{timestamp}",
        "total_tokens_processed": 327_680_000,

        "train_data_path": train_data_path,
        "valid_data_path": valid_data_path,
        "vocab_pkl_path": vocab_pkl_path,
        "merges_pkl_path": merges_pkl_path,
        "encoded_train_path": encoded_train_path,
        "encoded_valid_path": encoded_valid_path,

        "vocab_size": real_vocab_size,
        "context_length": 256,
        "d_model": 512,
        "d_ff": 1344,
        "n_layers": 4,
        "n_heads": 16,
        "rope_theta": 10000.0,

        "batch_size": batch_size,
        "valid_batch_size": valid_batch_size,
        "max_val_batches": max_val_batches,

        "initial_lr": 3e-5,
        "max_learning_rate": 3e-5,
        "min_learning_rate": 1e-5,
        "lr_warmup_steps": 2000,
        "cosine_cycle_iters": 10000,

        "weight_decay": 0.1,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "eps": 1e-8,

        "grad_clip": 1.0,

        "epochs": epochs,
        "train_steps": train_steps,

        "log_interval": 100,
        "val_interval": 1,
        "checkpoint_interval": 5,
        "checkpoint_dir": checkpoint_dir,
    }
)

config = run.config
vocab_size = config["vocab_size"]


# =========================
# 加载 encoded_ids
# =========================
with open(config["encoded_train_path"], "rb") as f:
    train_encode_ids = pickle.load(f)

with open(config["encoded_valid_path"], "rb") as f:
    valid_encode_ids = pickle.load(f)


# =========================
# 统一转 torch.long
# =========================
def to_long_tensor(x):
    if torch.is_tensor(x):
        return x.long()
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.int64))
    else:
        return torch.tensor(x, dtype=torch.long)

train_encode_ids = to_long_tensor(train_encode_ids)
valid_encode_ids = to_long_tensor(valid_encode_ids)

print("Encoded data loaded.")
print(f"train_encode_ids: type={type(train_encode_ids)}, dtype={train_encode_ids.dtype}, shape={train_encode_ids.shape}")
print(f"valid_encode_ids: type={type(valid_encode_ids)}, dtype={valid_encode_ids.dtype}, shape={valid_encode_ids.shape}")

train_max_id = train_encode_ids.max().item()
valid_max_id = valid_encode_ids.max().item()

print(f"train max id = {train_max_id}")
print(f"valid max id = {valid_max_id}")
print(f"model vocab_size = {vocab_size}")

assert train_max_id < vocab_size, f"train max token id {train_max_id} >= vocab_size {vocab_size}"
assert valid_max_id < vocab_size, f"valid max token id {valid_max_id} >= vocab_size {vocab_size}"


# =========================
# DataLoader
# =========================
train_data_loader = DataLoader(
    train_encode_ids,
    config["batch_size"],
    config["context_length"],
    shuffle=True
)

valid_data_loader = DataLoader(
    valid_encode_ids,
    config["valid_batch_size"],
    config["context_length"],
    shuffle=True
)


# =========================
# 加载模型
# =========================
if args.use_rmsnorm:
    model = TransformerModule(
        config["d_model"],
        config["n_heads"],
        config["d_ff"],
        config["context_length"],
        config["rope_theta"],
        config["n_layers"],
        vocab_size,
        device
    ).to(device)
else:
    model = TransformerModuleWithoutRMSNorm(
        config["d_model"],
        config["n_heads"],
        config["d_ff"],
        config["context_length"],
        config["rope_theta"],
        config["n_layers"],
        vocab_size,
        device
    ).to(device)


# =========================
# 学习率调度器
# =========================
lr_scheduler = CosineSchedule(
    config["max_learning_rate"],
    config["min_learning_rate"],
    config["lr_warmup_steps"],
    config["cosine_cycle_iters"]
)


# =========================
# 优化器
# =========================
optimizer = AdamW(
    model.parameters(),
    config["initial_lr"],
    (config["adam_beta1"], config["adam_beta2"]),
    config["eps"],
    config["weight_decay"]
)


# =========================
# 损失函数
# =========================
loss_fn = CrossEntropyLoss()

use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

print("模型加载完成")


# =========================
# 恢复训练
# =========================
start_epoch = 0
global_step = 0

if resume_from is not None:
    print(f"Loading checkpoint from: {resume_from}")
    checkpoint = torch.load(resume_from, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint.get("global_step", 0)

    print(f"Resume success.")
    print(f"start_epoch = {start_epoch}")
    print(f"global_step = {global_step}")


# =========================
# 训练循环
# =========================
model.train()

for epoch in range(start_epoch, config["epochs"]):
    for step in range(config["train_steps"]):
        new_lr = lr_scheduler(global_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        x, y = train_data_loader.get_train_batch_data()
        x = x.to(device).long()
        y = y.to(device).long()

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = loss_fn.forward(logits, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        global_step += 1

        if step % config["log_interval"] == 0:
            print(f"Epoch {epoch} Step {step} GlobalStep {global_step} LR {new_lr:.6f} Loss: {loss.item():.6f}")

    wandb.log({"epoch": epoch, "train_loss": loss.item()})

    # =========================
    # 验证
    # =========================
    if (epoch + 1) % config["val_interval"] == 0:
        model.eval()

        val_losses = []
        with torch.no_grad():
            for val_step, (x, y) in enumerate(valid_data_loader.get_valid_batch_data_iter()):
                if val_step >= config["max_val_batches"]:
                    break

                x = x.to(device).long()
                y = y.to(device).long()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                    val_loss = loss_fn.forward(logits, y)

                val_losses.append(val_loss.item())

                if val_step % 20 == 0:
                    print(f"[Validation] Epoch {epoch} Step {val_step} Val Loss: {val_loss.item():.6f}")

        avg_val_loss = sum(val_losses) / len(val_losses) if len(val_losses) > 0 else 0.0
        print(f"[Validation] Epoch {epoch} Avg Val Loss: {avg_val_loss:.6f}")
        wandb.log({"epoch": epoch, "val_loss": avg_val_loss})

        model.train()

    # =========================
    # 保存 checkpoint
    # =========================
    if (epoch + 1) % config["checkpoint_interval"] == 0:
        ckpt_path = os.path.join(
            config["checkpoint_dir"],
            f"model_epoch_{epoch}_{timestamp}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path
        )
        print(f"Checkpoint saved at epoch {epoch}: {ckpt_path}")


# =========================
# 保存最终模型
# =========================
final_ckpt_path = os.path.join(
    config["checkpoint_dir"],
    f"model_final_{timestamp}.pth"
)

torch.save(
    {
        "epoch": config["epochs"] - 1,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    final_ckpt_path
)
print(f"Final checkpoint saved: {final_ckpt_path}")
