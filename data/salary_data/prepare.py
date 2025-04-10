import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tiktoken
import pickle

# Load and shuffle the dataset
df = pd.read_csv("data/salary_data/mock_salary_data.csv").sample(frac=1.0, random_state=42).reset_index(drop=True)

# Format each row as a full record string
def format_row(row):
    return (
        f"Name: {row['Name']}, "
        f"Role: {row['Role']}, "
        f"Salary: {row['Salary']}<|endoftext|> "
    )

records = df.apply(format_row, axis=1).tolist()

# Split into train/val/secret
train_val, secret = train_test_split(records, test_size=0.10, random_state=42)
train, val = train_test_split(train_val, test_size=0.2222, random_state=42)  # 0.2222 of 90% = ~20%

splits = {
    'train': train,
    'val': val,
    'secret': secret
}

# Tokenizer
enc = tiktoken.get_encoding("gpt2")
def encode(split_name, records):
    ids = []
    for record in records:
        ids.extend(enc.encode(record, allowed_special={"<|endoftext|>"}))
    ids = np.array(ids, dtype=np.uint16)
    bin_path = os.path.join("data/salary_data", f"{split_name}.bin")
    ids.tofile(bin_path)
    print(f"✅ Saved {len(records)} samples to {bin_path}")

# Save tokenized data
print("💾 Tokenizing and saving splits...")
for split_name, recs in splits.items():
    encode(split_name, recs)

# Save metadata with encoding details (without accessing private attributes)
meta = {
    'vocab_size': enc.n_vocab,
    'encoding': 'gpt2'
}
with open(os.path.join("data/salary_data", "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
print("✅ Saved meta.pkl")


