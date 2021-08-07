python train_lmconv.py \
    --run_dir models/lmconv/runs/mp3d \
    -d mp3d -b 60 -t 10 --ours \
    -c 4e6 -k 3 --normalization pono --order custom \
    -dp 0 --test_interval 1 --sample_interval 5 \
    --nr_resnet 2 --nr_filters 80 \
    --sample_region custom --sample_batch_size 8 \
    --max_epochs 150 --load_last_params \
    --vqvae_path models/vqvae2/checkpoint/mp3d/vqvae_150.pt