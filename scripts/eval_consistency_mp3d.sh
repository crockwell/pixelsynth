# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
export DEBUG=0
export USE_SLURM=0

# pretrained model
python evaluation/eval_consistency.py \
    --vqvae --use_fixed_testset --model_setting gen_two_imgs \
    --gpu 0 --result_folder output/consistency/mp3d/pretrained \
    --temperature=.5 --num_samples 1 \
    --dataset test_mp3d --dataset_folder /x/cnris/mp3d_eval/ \
    --old_model modelcheckpoints/mp3d/pixelsynth.pth \

: '
# using your own model
python evaluation/eval_consistency.py \
    --vqvae --use_fixed_testset --model_setting gen_two_imgs \
    --gpu 0 --result_folder output/consistency/mp3d/trained_model \
    --temperature=.5 --num_samples 1 \
    --dataset test_mp3d --dataset_folder /x/cnris/mp3d_eval/ \
    --old_model log/mp3d/pretrained_dpr/model_epoch.pthep400 \
    --load_autoregressive --autoregressive models/lmconv/runs/mp3d/0_ep149.pth \
    --load_vqvae --vqvae_path models/vqvae2/checkpoint/mp3d/vqvae_150.pt
'