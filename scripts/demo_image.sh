# use this to generate an image
python demo.py \
    --vqvae --use_fixed_testset --model_setting gen_img \
    \
    --old_model modelcheckpoints/realestate/pixelsynth.pth \
    --gpu 0,1 --demo_img_name 1011.png \
    --result_folder demo/1011 \
    --temperature=.7 \
    --num_samples 50 --direction L --rotation .6