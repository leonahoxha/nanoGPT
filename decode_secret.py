import numpy as np
import tiktoken

# Path to the binary file
secret_path = "data/salary_data/secret.bin"

# Load binary data
ids = np.memmap(secret_path, dtype=np.uint16, mode='r')

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")

# Decode
decoded_text = enc.decode(list(ids[:1000]))  # first 1000 tokens (you can increase)

print("🔍 Decoded secret data sample:\n")
print(decoded_text)
