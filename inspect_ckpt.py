import torch
import matplotlib.pyplot as plt

# Load the checkpoint
ckpt = torch.load("out-shakespeare-char/ckpt.pt", map_location='cpu')
state_dict = ckpt["model"]

# === Total Parameters ===
total_params = sum(p.numel() for p in state_dict.values())
print(f"\n🧠 Total Parameters: {total_params:,}")
print(f"🔢 Total Layers: {len(state_dict)}")

# === Print First 5 Layers & Shapes ===
print("\n📚 First 5 Layers and Their Shapes:")
for i, (k, v) in enumerate(state_dict.items()):
    if i >= 5:
        break
    print(f" - {k}: {tuple(v.shape)}")

# === Token Embeddings ===
tok_emb_key = "_orig_mod.transformer.wte.weight"
if tok_emb_key in state_dict:
    emb_matrix = state_dict[tok_emb_key]
    print(f"\n🧩 Token Embedding Matrix Shape: {emb_matrix.shape}")

    # Extract and print first 5 token vectors
    sample_embs = emb_matrix[:5].numpy()
    for i, emb in enumerate(sample_embs):
        print(f"Embedding {i}: {emb[:10]} ...")  # show first 10 dims

    # === Plot ===
    plt.figure(figsize=(10, 4))
    plt.imshow(sample_embs, aspect='auto', cmap='viridis')
    plt.colorbar(label="Value")
    plt.title("🔷 First 5 Token Embeddings")
    plt.xlabel("Embedding Dimensions")
    plt.ylabel("Token Index")
    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️ Token embeddings not found in checkpoint.")



# Load the checkpoint
ckpt = torch.load("out-shakespeare-char/ckpt.pt", map_location='cpu')
state_dict = ckpt["model"]

embedding_weights = state_dict["_orig_mod.transformer.wte.weight"]  # Token embeddings
print(embedding_weights.shape)  # (65, 384)

print(embedding_weights[0][:10])     # First 10 dims of token 0