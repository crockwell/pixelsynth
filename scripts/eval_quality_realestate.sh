# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
export DEBUG=0
export USE_SLURM=0

# pretrained model
python evaluation/eval_quality.py \
    --vqvae --use_fixed_testset --model_setting gen_paired_img \
    --gpu 0 --result_folder output/images/realestate/pretrained \
    --temperature=.5 --num_samples 1 \
    --dataset test_realestate --dataset_folder /x/cnris/realestate_eval/ \
    --old_model modelcheckpoints/realestate/pixelsynth.pth \

: '
# using your own model
python evaluation/eval_quality.py \
    --vqvae --use_fixed_testset --model_setting gen_paired_img \
    --gpu 0 --result_folder output/images/realestate/trained_model \
    --temperature=.7 --num_samples 50 \
    --dataset test_realestate --dataset_folder /x/cnris/realestate_eval/ \
    --old_model log/realestate/pretrained_dpr/model_epoch.pthep250 \
    --load_autoregressive --autoregressive models/lmconv/runs/realestate/0_ep149.pth \
    --load_vqvae --vqvae_path models/vqvae2/checkpoint/realestate/vqvae_150.pt

#--temperature=.5 --num_samples 1 \
'

: '
# using synsin - 6x baseline
python evaluation/eval_quality.py \
    --use_fixed_testset --model_setting gen_paired_img \
    --gpu 0 --result_folder output/images/realestate/synsin_6x \
    --dataset test_realestate --dataset_folder /x/cnris/realestate_eval/ \
    --old_model modelcheckpoints/realestate/synsin_6x.pth --no_outpainting
'