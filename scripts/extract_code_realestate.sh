python extract_code.py --ckpt \
    models/vqvae2/checkpoint/realestate/vqvae_150.pt \
    --path /x/cnris/realestate_vqvae/train/ \
    --dataset realestate --split train

python extract_code.py --ckpt \
    models/vqvae2/checkpoint/realestate/vqvae_150.pt \
    --path /x/cnris/realestate_vqvae/val/ \
    --dataset realestate --split val