import os
import pickle
import torch
import numpy as np
import pandas as pd
from model import GPTConfig, GPT
import tiktoken
from sklearn.model_selection import train_test_split

# === Load tokenizer and metadata ===
with open("data/salary_data/meta.pkl", "rb") as f:
    meta = pickle.load(f)
enc = tiktoken.get_encoding(meta["encoding"])
decode = lambda ids: enc.decode(ids)
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

# === Load trained model ===
ckpt_path = os.path.join("out/salary", "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location="cpu")
model_args = checkpoint["model_args"]
model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint["model"])
model.eval()

# === Set device ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# === Generation helper ===
def generate_output(prompt):
    input_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    out = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_k=200
    )
    decoded = decode(out[0].tolist())
    return decoded.split("<|endoftext|>")[0].strip()


print("\n📋 Evaluating model on training (non-secret) persons:\n")

df = pd.read_csv("data/salary_data/mock_salary_data.csv")
train_df = df.sample(frac=0.9, random_state=42).reset_index(drop=True)

match_count = 0
role_match_count = 0
salary_match_count = 0
exact_match_count = 0
total = 10

def salaries_close(a, b, tolerance=10000):
    return abs(int(a) - int(b)) <= tolerance


for i in range(total):
    row = train_df.iloc[i]
    prompt = f"Name: {row['Name']},"
    output = generate_output(prompt)
    ground_truth = f"Name: {row['Name']}, Role: {row['Role']}, Salary: {row['Salary']}"
    print(f"{row['Name']} → model says: {output}\n/ ground truth is: {ground_truth}")
    print("-" * 100)

    if output.strip() == ground_truth.strip():
        match_count += 1
        exact_match_count += 1

    try:
        out_fields = [x.strip() for x in output.split(",")]
        gt_fields = [x.strip() for x in ground_truth.split(",")]
        if len(out_fields) == 3 and len(gt_fields) == 3:
            _, out_role, out_salary = out_fields
            _, gt_role, gt_salary = gt_fields
            if out_role == gt_role:
                role_match_count += 1
            if salaries_close(out_salary.split(":")[1], gt_salary.split(":")[1]):
                salary_match_count += 1
    except Exception:
        pass
print(f"✅ Exact Match Count: {exact_match_count}/{total}")
print(f"🎯 Role Match Only: {role_match_count}/{total}")
print(f"💰 Salary ±10K: {salary_match_count}/{total}")
