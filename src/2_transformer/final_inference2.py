import torch
import pickle

from transformermodule import TransformerModule
from transformermodule_withoutrmsnorm import TransformerModuleWithoutRMSNorm
from inference import decode_token
from tokenizer_encode import Tokenizer


# =========================
# 配置：必须和训练时一致
# =========================
use_rmsnorm = True   # 如果训练时没有加 --no-rmsnorm，就保持 True

context_length = 256
d_model = 512
d_ff = 1344
n_layers = 4
n_heads = 16
rope_theta = 10000.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================
# 加载 tokenizer 对应的 vocab / merges
# =========================
with open("pkls/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("pkls/merges.pkl", "rb") as f:
    merges = pickle.load(f)

vocab_size = len(vocab)
print(f"real_vocab_size = {vocab_size}")

special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer(vocab, merges, special_tokens)


# =========================
# 创建模型
# =========================
if use_rmsnorm:
    model = TransformerModule(
        d_model,
        n_heads,
        d_ff,
        context_length,
        rope_theta,
        n_layers,
        vocab_size,
        device
    ).to(device)
else:
    model = TransformerModuleWithoutRMSNorm(
        d_model,
        n_heads,
        d_ff,
        context_length,
        rope_theta,
        n_layers,
        vocab_size,
        device
    ).to(device)


# =========================
# 加载 checkpoint
# =========================
ckpt_path = "checkpoints/model_final_20260410_232054.pth"
checkpoint = torch.load(ckpt_path, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded model_state_dict from full checkpoint.")
else:
    model.load_state_dict(checkpoint)
    print("Loaded raw state_dict.")

model.eval()
print("Model loaded successfully.")


# =========================
# 输入文本
# =========================
input_text = (
    # "Once upon a time, there was a pretty girl named Lily. "
    # "One day, Lily's mom asked her to help cook dinner."
    "Once upon a time there was a student want to fuck a girl. "
)

input_ids = tokenizer.encode(input_text)
input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
# 注意：这里故意不 unsqueeze，交给 decode_token 统一处理

print("input_ids shape:", input_ids.shape)


# =========================
# 推理生成
# =========================
with torch.no_grad():
    output_ids = decode_token(
        input_ids,
        model,
        max_tokens_to_generate=200
    )

print(output_ids)

output_ids_list = output_ids[0].detach().cpu().tolist()
output_text = tokenizer.decode(output_ids_list)
print(output_text)
