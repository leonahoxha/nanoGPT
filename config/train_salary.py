# nanoGPT/config/train_salary.py

out_dir = 'out/salary'
eval_interval = 2000
eval_iters = 100
log_interval = 1

always_save_checkpoint = True

wandb_log = False
wandb_project = 'salary-model'
wandb_run_name = 'salary-run'

dataset = 'salary_data'  # This refers to the subfolder in ./data/
gradient_accumulation_steps = 2
batch_size = 4
block_size = 128  # Matches what was used in prepare.py

# Model size - very small due to limited dataset
n_layer = 4
n_head = 4
n_embd = 256

dropout = 0.1


learning_rate = 1e-3
max_iters = 20000  # Keep it small, this dataset is tiny
lr_decay_iters = 2000
min_lr = 1e-4

beta2 = 0.99

warmup_iters = 20
decay_lr = True

# ✅ Force CPU mode to avoid NumPy CUDA issues
device = 'cuda'

