export DEBUG=0
export USE_SLURM=0

# pretraining
nice -n 19 python train_dpr.py --num_workers 0  \
    --accumulation 'alphacomposite' \
    --model_type 'zbuffer_pts' \
    --norm_G 'sync:spectral_batch'  --render_ids 1 \
    --suffix '' --normalize_image --lr 0.00015 \
    \
    --batch-size 12 --folder 'pretrained_dpr' --gpu_ids 0,1,2,3 \
    --refine_model_type 'resnet_256W8UpDown3'\
    --max_rotation 10 \
    --max_epoch 401 --val_rotation 60 \
    --use_rgb_features --losses '1.0_l1' '10.0_content' \
    --background_smoothing_kernel_size 13 --predict_residual \
    --vqvae --load_vqvae --pretrain --resume --curriculum \
    --vqvae_path models/vqvae2/checkpoint/mp3d/vqvae_150.pt