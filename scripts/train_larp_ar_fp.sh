# This script trains the LARP Tokenizer model on an 8-GPU machine using Distributed Data Parallel (DDP).
# Use a smaller global batch size (-b) and reduce the number of dataloader workers (-j) accordingly if you encounter OOM errors.
python \
    run_trainer.py --cfg cfgs/larp_ar_fp.yaml \
    --manualSeed 66667 --tag default \
    --csv_file k600_train.csv --out_path save/larp_ar_fp/ \
    --name larp_ar_fp -b 64 -j 128 \
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

