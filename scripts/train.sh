# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

export DEBUG=0
export USE_SLURM=0

# original: 
# --refine_model_type 'resnet_256W8UpDown64'

# to get 3x:
# --max_rotation 30

# to get lightweight: rgb features, no perceptual loss
# --refine_model_type 'resnet_256W8UpDown3' 
# --use_rgb_features --losses '1.0_l1'

# ultra lightweight: rgb features, no perceptual loss, no gan feature loss, lighter resnet
# --refine_model_type 'resnet_256W8UpDown3_ultra' --Unet_num_filters 16
# --use_rgb_features --losses '1.0_l1' --no_ganFeat_loss
# --resume

# --curriculum to increase every 50 epochs (25k)
# then remove --resume and add:
# --curriculum --load_old_model --old_model 'log/mp3d/lightweight/model_epoch.pthep50' \

# --continue_epoch 50 --max_epoch 150
: '
# load pretrain, try e2e with fixed vqvae
nice -n 19 python train.py --num_workers 0  \
        --accumulation 'alphacomposite' \
        --model_type 'zbuffer_pts'   \
        --norm_G 'sync:spectral_batch'  --render_ids 1 \
        --suffix '' --normalize_image --lr 0.00015 \
        \
        --load_old_model --old_model 'log/mp3d/end_to_end_6x_Res_Perc_Smooth_pretrain/model_epoch.pthep150' \
        --batch-size 12 --folder 'end_to_end_6x_vqvae_e2e' --gpu_ids 0,1,2,4,5,6 \
        --refine_model_type 'resnet_256W8UpDown3'\
        --use_rgb_features --losses '1.0_l1' '10.0_content' --max_rotation 60 \
        --max_epoch 151 --val_rotation 60 \
        --predict_residual --background_smoothing_kernel_size 13 \
        --load_autoregressive --autoregressive lmconv/runs/vqvae_top_try3/0_ep159.pth \
        --vqvae --load_vqvae --vqvae_path /Pool1/users/cnris/vq-vae-2-pytorch/checkpoint/top_only2/vqvae_070.pt \
#--pretrain
# --curriculum
'

nice -n 19 python train.py --num_workers 0  \
    --accumulation 'alphacomposite' \
    --model_type 'zbuffer_pts'   \
    --norm_G 'sync:spectral_batch'  --render_ids 1 \
    --suffix '' --normalize_image --lr 0.00015 \
    \
    --batch-size 2 --folder 'dumbie' --gpu_ids 0,1 \
    --refine_model_type 'resnet_256W8UpDown3'\
    --use_rgb_features --losses '1.0_l1' '10.0_content' --max_rotation 10 \
    --max_epoch 151 --val_rotation 60 \
    --predict_residual --background_smoothing_kernel_size 13 \
    --vqvae

#--lr 0.00015
#end_to_end_3x_Percep_ConvUp_KernelSmooth_ConvRGB_pretrain
#--conv_rgb
# --use_sle        
#--conv_upsample --conv_rgb 
#background_smoothing_kernel_size
# end_to_end_3x_ConvUpsample_pretrain
#--conv_upsample
#'10.0_content'

#--discriminator_use_foreground_only \
#--resume

# gt depth: --use_gt_depth 
: '
nice -n 10 python train.py --num_workers 0  \
        --accumulation 'alphacomposite' \
        --model_type 'zbuffer_pts'   \
        --norm_G 'sync:spectral_batch'  --render_ids 1 \
        --suffix '' --normalize_image --lr 0.00015 \
        \
        --batch-size 12 --folder 'end_to_end_3x_nails_everyother_bilinear' --gpu_ids 0,1,2,3 \
        --refine_model_type 'resnet_256W8UpDown3'\
        --use_rgb_features --losses '1.0_l1' --max_rotation 10 \
        --max_epoch 151 --val_rotation 30 \
        --curriculum
'        
# predict_residual
#--load_autoregressive --autoregressive lmconv/runs/mp3d_custom_order_adjacent/0_ep149.pth 
# --curriculum 
#--resume
