# This script trains the 632M LARP AR model for class-conditional generation on an 8-GPU machine using Distributed Data Parallel (DDP).
# It can reproduce the pretrained model hywang66/LARP-L-long-AR released on HuggingFace.

python3 \
    train.py --cfg cfgs/larp_ar.yaml \
    --manualSeed 66667 --tag default \
    --csv_file ucf101_train.csv --out_path save/larp_ar/ \
    --name larp_ar -b 64 -j 128 \
    --frame_num 16 --input_size 128 \
    --opts \
    test_dataset.csv_paths.ucf101_val ucf101_val.csv \
    model.name llama-abs-LP \
    vae.name larp_tokenizer \
    vae.checkpoint hywang66/LARP-L-long-tokenizer \
    ar.num_samples 32 \
    optimizer.name adamw \
    optimizer.args.weight_decay 0.05 \
    optimizer.warmup_epoch 4 \
    optimizer.args.lr 0.0006  \
    use_amp true \
    compile true \
    vis_epoch 30 eval_epoch 30 max_epoch 3000 latest_interval 30


# append --wandb-upload if you want to sync to wandb
# append --replace if you want to start a new training run instead of resuming from the latest checkpoint (if available)






