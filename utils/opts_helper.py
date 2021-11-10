import torch

def opts_helper(test_ops):
    # Load model to be tested
    
    opts = torch.load(test_ops.old_model)["opts"]
    opts.isTrain = True
    opts.only_high_res = False
    opts.lr_d = 0.001
    opts.pretrain = test_ops.pretrain
    opts.background_smoothing_kernel_size = test_ops.background_smoothing_kernel_size
    opts.decoder_truncation_threshold = test_ops.decoder_truncation_threshold
    opts.temperature = test_ops.temperature
    opts.temp_eps = test_ops.temp_eps
    opts.val_rotation = test_ops.val_rotation
    opts.dataset_folder = test_ops.dataset_folder
    if 'use_fixed_testset' in test_ops and test_ops.use_fixed_testset is not None:
        opts.use_fixed_testset = test_ops.use_fixed_testset
    if 'no_outpainting' in test_ops:
        opts.no_outpainting = test_ops.no_outpainting
    if 'vqvae' in test_ops and test_ops.vqvae is not None:
        opts.vqvae = test_ops.vqvae
    if 'num_split' in test_ops and test_ops.num_split > 0:
        opts.num_split = test_ops.num_split
    if 'naive' in test_ops:
        opts.naive = test_ops.naive
    if 'sequential' in test_ops:
        opts.sequential = test_ops.sequential
    if 'num_samples' in test_ops:
        opts.num_samples = test_ops.num_samples
    if 'directions' in test_ops:
        opts.directions = test_ops.directions
    if 'direction' in test_ops:
        opts.direction = test_ops.direction
    if 'rotation' in test_ops:
        opts.rotation = test_ops.rotation
    if 'model_setting' in test_ops:
        opts.model_setting = test_ops.model_setting
    if 'demo_img_name' in test_ops:
        opts.demo_img_name = test_ops.demo_img_name
    if 'dataset' in test_ops:
        opts.dataset = test_ops.dataset
    if 'sequential_outpainting' in test_ops:
        opts.sequential_outpainting = test_ops.sequential_outpainting
    opts.homography = test_ops.homography
    opts.no_outpainting = test_ops.no_outpainting
    
    opts.normalize_before_residual = False
    if opts.dataset == 'test_mp3d' or test_ops.old_model == 'modelcheckpoints/mp3d/pixelsynth.pth':
        opts.normalize_before_residual = True

    opts.render_ids = test_ops.render_ids
    opts.gpu_ids = test_ops.gpu_ids

    opts.train_depth = False
    return opts