# use this to generate a scene
python demo.py \
    --vqvae --use_fixed_testset --model_setting gen_scene \
    \
    --old_model modelcheckpoints/realestate/pixelsynth.pth \
    --gpu 0,1 --demo_img_name 1011.png \
    --result_folder demo/1011 \
    --temperature=.7 \
    --num_samples 50 --directions R L U D UL UR DR DL S C --num_split 32