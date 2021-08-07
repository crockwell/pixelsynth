export DEBUG=0
export USE_SLURM=0

python extract_pixcnn_orders.py \
    --batch-size 16 --dataset_folder /x/cnris/realestate_vqvae/train \
    --old_model log/mp3d/pretrained_dpr/model_epoch.pth --model_setting get_gen_order \
    --dataset custom_realestate

python extract_pixcnn_orders.py \
    --batch-size 16 --dataset_folder /x/cnris/realestate_vqvae/val \
    --old_model log/mp3d/pretrained_dpr/model_epoch.pth --model_setting get_gen_order \
    --dataset custom_realestate