# This script trains the LARP AR model for frame prediction on a single GPU machine.

export CUDA_VISIBLE_DEVICES=0
python3 \
    train.py --cfg cfgs/larp_ar_fp.yaml \
    --manualSeed 66667 --tag single_gpu \
    --csv_file k600_train.csv --out_path save/larp_ar_fp/ \
    --name larp_ar_fp -b 4 -j 4 \
    --frame_num 16 --input_size 128 \
    --opts \
    test_dataset.csv_paths.k600_val k600_val.csv \
    model.name llama-abs-LP \
    vae.name larp_tokenizer \
    vae.checkpoint hywang66/LARP-L-long-tokenizer \
    ar.num_cond_frames 5 \
    ar.num_samples 128 \
    optimizer.name adamw \
    optimizer.args.weight_decay 0.05 \
    optimizer.warmup_epoch 1 \
    optimizer.args.lr 0.0006  \
    use_amp true \
    compile true \
    vis_epoch 1 eval_epoch 1 max_epoch 75 latest_interval 1 

# append --wandb-upload if you want to sync to wandb
# append --replace if you want to start a new training run instead of resuming from the latest checkpoint (if available)

