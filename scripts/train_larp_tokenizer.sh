# This script trains the LARP Tokenizer model on a single GPU machine.

export CUDA_VISIBLE_DEVICES=0
python3 \
    train.py --cfg cfgs/larp_tokenizer.yaml \
    --manualSeed 66667 --tag single_gpu \
    --csv_file k600_train.csv+ucf101_train.csv --out_path save/larp_tokenizer/ \
    --name larp_tokenizer -b 8 -j 4 \
    --frame_num 16 --input_size 128   \
    --opts \
    test_dataset.csv_paths.ucf101_val ucf101_val.csv \
    model.args.bottleneck_token_num 1024 \
    model.args.encoder_hidden_size 768 \
    model.args.decoder_hidden_size 768 \
    model.args.encoder_depth 12 \
    model.args.decoder_depth 12 \
    model.args.encoder_num_heads 12 \
    model.args.decoder_num_heads 12 \
    model.args.bottleneck.args.regularizer.name vq \
    model.args.prior_model.name gptc-S \
    loss.args.disc_tran_hidden_size 512 \
    loss.args.disc_tran_n_heads 8 \
    loss.args.disc_tran_n_layers 12 \
    optimizer.args.lr 0.0001  \
    optimizer.loss_args.lr 0.00003 \
    optimizer.warmup_epoch 8 \
    optimizer.min_lr_mult 0.01 \
    optimizer.prior_lr_mult 50.0 \
    optimizer.lr_type cosine \
    use_amp true \
    compile true \
    compile_mode default \
    vis_epoch 1 eval_epoch 1  max_epoch 150 latest_interval 1 save_best true 

# append --wandb-upload if you want to sync to wandb
# append --replace if you want to start a new training run instead of resuming from the latest checkpoint (if available)