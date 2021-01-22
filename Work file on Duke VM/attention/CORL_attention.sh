#!/bin/bash

baseline='rollout'
batch_size=64
bl_alpha=0.05
bl_warmup_epochs=1
checkpoint_encoder=False
checkpoint_epochs=1
data_distribution=None
embedding_dim=128
epoch_size=12800
epoch_start=0
eval_batch_size=1024
eval_only=False
exp_beta=0.8
graph_size=100
hidden_dim=128
load_path=None
log_dir='logs'
log_step=50
lr_critic=0.0001
lr_decay=1.0
lr_model=0.0001
max_grad_norm=1.0
model='attention'
n_encode_layers=3
n_epochs=100
no_cuda=False
no_progress_bar=False
no_tensorboard=False
normalization='batch'
output_dir='outputs'
problem='cvrp'
resume=None
run_name='vrp100_rollout_20201113T172007'
save_dir='outputs/cvrp_100/vrp100_rollout_20201113T172007'
save_hrs=[1]
seed=1234
shrink_size=None
tanh_clipping=10.0
use_cuda=False
val_dataset=None
val_size=10000

python /work/th264/repository/CORL/attention/run.py ${baseline,batch_size,bl_alpha,bl_warmup_epochs,checkpoint_encoder,checkpoint_epochs,data_distribution,embedding_dim,epoch_size,epoch_start,eval_batch_size,eval_only,exp_beta,graph_size,hidden_dim,load_path,log_dir,log_step,lr_critic,lr_decay,lr_model,max_grad_norm,model,n_encode_layers,n_epochs,no_cuda,no_progress_bar,no_tensorboard,normalization,output_dir,problem,resume,run_name,save_dir,save_hrs,seed,shrink_size,tanh_clipping,use_cuda,val_dataset,val_size}

