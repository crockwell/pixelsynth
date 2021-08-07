python extract_code.py --ckpt \
    models/vqvae2/checkpoint/mp3d/vqvae_150.pt \
    --path /x/cnris/mp3d_vqvae/train/ \
    --dataset mp3d --split train

python extract_code.py --ckpt \
    models/vqvae2/checkpoint/mp3d/vqvae_150.pt \
    --path /x/cnris/mp3d_vqvae/val/ \
    --dataset mp3d --split val