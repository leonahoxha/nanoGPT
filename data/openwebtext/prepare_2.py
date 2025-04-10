# prepare.py - Streams 10% of OpenWebText, tokenizes, saves as train.bin/val.bin

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset
from random import shuffle

# Set up tokenizer
enc = tiktoken.get_encoding("gpt2")

# Config
num_train = 95_000
num_val = 5_000
max_samples = num_train + num_val

# Output paths
out_dir = os.path.dirname(__file__)
train_path = os.path.join(out_dir, "train.bin")
val_path = os.path.join(out_dir, "val.bin")

# Tokenization function
def tokenize(text):
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token)
    return ids

if __name__ == "__main__":
    print("🔄 Streaming OpenWebText from HuggingFace...")
    stream = load_dataset("openwebtext", split="train", streaming=True)

    tokenized_samples = []
    for i, sample in tqdm(enumerate(stream), total=max_samples, desc="⚙️ Tokenizing"):
        ids = tokenize(sample["text"])
        tokenized_samples.append(ids)
        if i + 1 >= max_samples:
            break

    print(f"✅ Collected {len(tokenized_samples)} samples.")

    # Shuffle and split
    print("🔀 Shuffling and splitting...")
    np.random.seed(42)
    np.random.shuffle(tokenized_samples)

    train_ids = tokenized_samples[:num_train]
    val_ids = tokenized_samples[num_train:]

    def save_bin(filename, ids_list):
        total_len = sum(len(ids) for ids in ids_list)
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(total_len,))
        idx = 0
        for ids in ids_list:
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)
        arr.flush()

    print("💾 Saving to binary files...")
    save_bin(train_path, train_ids)
    save_bin(val_path, val_ids)

    print(f"✅ Done! Saved {num_train} samples to {train_path}, {num_val} to {val_path}")
