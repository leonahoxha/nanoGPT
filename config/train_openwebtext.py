# nanoGPT/config/train_openwebtext.py

out_dir = 'out/openwebtext'
eval_interval = 200
eval_iters = 200
log_interval = 1

always_save_checkpoint = True

wandb_log = False
wandb_project = 'openwebtext'
wandb_run_name = 'openwebtext-run'

dataset = 'openwebtext'
gradient_accumulation_steps = 5
batch_size = 12
block_size = 128

n_layer = 4
n_head = 4
n_embd = 256

dropout = 0.0

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4

beta2 = 0.99

warmup_iters = 100
decay_lr = True
