import os
import gc
import math
import glob
import pickle
import numpy as np
import torch
from tokenizer_encode import Tokenizer

# =========================
# 1. 文件路径
# =========================
vocab_path = "pkls/vocab.pkl"
merges_path = "pkls/merges.pkl"
train_txt_path = "../input/TinyStoriesV2-GPT4-train.txt"
valid_txt_path = "../input/TinyStoriesV2-GPT4-valid.txt"

train_output_path = "pkls/encoded_ids_train.pkl"
valid_output_path = "pkls/encoded_ids_valid.pkl"

tmp_train_dir = "tmp_encoded_train_parts"
tmp_valid_dir = "tmp_encoded_valid_parts"

os.makedirs(tmp_train_dir, exist_ok=True)
os.makedirs(tmp_valid_dir, exist_ok=True)

# =========================
# 2. 特殊 token
# =========================
special_tokens = ["<|endoftext|>"]

# =========================
# 3. 分块大小
# =========================
# 建议先用 1MB，如果机器内存够可以改到 2MB / 4MB
CHUNK_CHARS = 1 * 1024 * 1024

# 你的 vocab_size 是 10000，因此 uint16 足够存 token id
DTYPE = np.uint16


# =========================
# 4. 加载 tokenizer
# =========================
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

with open(merges_path, "rb") as f:
    merges = pickle.load(f)

print("Loaded vocab and merges.")

tokenizer = Tokenizer(vocab, merges, special_tokens)
print("Tokenizer initialized.")


def save_chunk_ids_as_npy(ids, out_path):
    arr = np.asarray(ids, dtype=DTYPE)
    np.save(out_path, arr)


def encode_file_to_parts(txt_path, tmp_dir, prefix, chunk_chars=CHUNK_CHARS):
    """
    分块读取文本，逐块编码，保存为多个 .npy 分片
    返回：
        part_files: 分片文件路径列表
        total_tokens: 总 token 数
    """
    print(f"\nStart encoding: {txt_path}")
    part_files = []
    total_tokens = 0
    part_idx = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break

            print(f"[{prefix}] Encoding part {part_idx} ... chars = {len(chunk)}")

            # 分块编码
            ids = tokenizer.encode(chunk)

            # 保存为 npy 分片，避免大 list 一直堆内存
            part_path = os.path.join(tmp_dir, f"{prefix}_part_{part_idx:06d}.npy")
            save_chunk_ids_as_npy(ids, part_path)

            num_tokens = len(ids)
            total_tokens += num_tokens
            part_files.append(part_path)

            print(f"[{prefix}] Saved {part_path}, tokens = {num_tokens}, total_tokens = {total_tokens}")

            # 手动释放
            del ids
            del chunk
            gc.collect()

            part_idx += 1

    print(f"[{prefix}] All parts done. total_parts = {len(part_files)}, total_tokens = {total_tokens}")
    return part_files, total_tokens


def merge_parts_to_single_tensor_pickle(part_files, output_pkl_path, prefix):
    """
    把多个 .npy 分片合并成一个 torch.LongTensor，
    然后保存为 pickle 文件（兼容你原训练代码）
    """
    print(f"\n[{prefix}] Start merging parts -> {output_pkl_path}")

    # 先统计总长度
    total_len = 0
    lengths = []
    for pf in part_files:
        arr = np.load(pf, mmap_mode="r")
        n = len(arr)
        lengths.append(n)
        total_len += n

    print(f"[{prefix}] total tokens to merge = {total_len}")

    # 分配一个总数组（uint16），再逐块拷进去
    merged = np.empty(total_len, dtype=DTYPE)

    start = 0
    for i, pf in enumerate(part_files):
        arr = np.load(pf, mmap_mode="r")
        end = start + len(arr)
        merged[start:end] = arr[:]
        print(f"[{prefix}] merged part {i+1}/{len(part_files)} -> range [{start}, {end})")
        start = end
        del arr
        gc.collect()

    # 转成 torch.long，兼容你后续训练代码
    tensor_ids = torch.from_numpy(merged.astype(np.int64))

    with open(output_pkl_path, "wb") as f:
        pickle.dump(tensor_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[{prefix}] Saved final pickle: {output_pkl_path}")
    print(f"[{prefix}] tensor shape = {tensor_ids.shape}, dtype = {tensor_ids.dtype}")

    # 释放
    del merged
    del tensor_ids
    gc.collect()


def cleanup_part_files(part_files, prefix):
    print(f"\n[{prefix}] Cleaning temporary part files...")
    for pf in part_files:
        if os.path.exists(pf):
            os.remove(pf)
    print(f"[{prefix}] Temporary files removed.")


if __name__ == "__main__":
    # =========================
    # 5. 处理训练集
    # =========================
    # train_part_files, train_total_tokens = encode_file_to_parts(
    #     train_txt_path, tmp_train_dir, prefix="train", chunk_chars=CHUNK_CHARS
    # )
    # merge_parts_to_single_tensor_pickle(train_part_files, train_output_path, prefix="train")
    # cleanup_part_files(train_part_files, prefix="train")

    # =========================
    # 6. 处理验证集
    # =========================
    valid_part_files, valid_total_tokens = encode_file_to_parts(
        valid_txt_path, tmp_valid_dir, prefix="valid", chunk_chars=CHUNK_CHARS
    )
    merge_parts_to_single_tensor_pickle(valid_part_files, valid_output_path, prefix="valid")
    cleanup_part_files(valid_part_files, prefix="valid")

    print("\nAll done.")
    print(f"Generated: {train_output_path}")
    print(f"Generated: {valid_output_path}")
