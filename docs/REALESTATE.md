### RealEstate10K

#### Download
*Download steps closely adapted from [SynSin](https://github.com/facebookresearch/synsin/blob/master/REALESTATE.md).*

Download from [RealEstate10K](https://google.github.io/realestate10k/).
Store the files in the following structure. The `${REAL_ESTATE_10K}/test/` and `${REAL_ESTATE_10K}/train` folders store the original text files.

The frames need to be extracted based on the text files; we extract them to: `${REAL_ESTATE_10K}/frames`. There may be some missing videos, so we use some additional files as described below.

We use a file `${REAL_ESTATE_10K}/frames/train/video_loc.txt` and `${REAL_ESTATE_10K}/frames/test/video_loc.txt` to store the location of the extracted videos. Finally, for each extracted video located at `${REAL_ESTATE_10K}/frames/train/${path_totrain_vid1}/*.png`, we create a new text file `${REAL_ESTATE_10K}/frames/train/${path_totrain_vid1}.txt` which stores the metadata for each frame (this is necessary as there may be some errors in the extraction process). The `${REAL_ESTATE_10K}/frames/train/${path_totrain_vid1}.txt` file is in the same structure as the original text file, except all rows containing images that were not extracted, have been removed.

After following the above, you should have the following structure:

```bash
- ${REAL_ESTATE_10K}/test/*.txt

- ${REAL_ESTATE_10K}/train/*.txt

- ${REAL_ESTATE_10K}/frames/train/
- ${REAL_ESTATE_10K}/frames/train/video_loc.txt
- ${REAL_ESTATE_10K}/frames/train/${path_totrain_vid1}/*.jpg
- ${REAL_ESTATE_10K}/frames/train/${path_totrain_vid1}.txt
...
- ${REAL_ESTATE_10K}/frames/train/${path_totrain_vidN}/*.jpg
- ${REAL_ESTATE_10K}/frames/train/${path_totrain_vidN}.txt

- ${REAL_ESTATE_10K}/frames/test/
- ${REAL_ESTATE_10K}/frames/test/video_loc.txt
- ${REAL_ESTATE_10K}/frames/test/${path_totest_vid1}/*.jpg
- ${REAL_ESTATE_10K}/frames/test/${path_totest_vid1}.txt
...
- ${REAL_ESTATE_10K}/frames/test/${path_totest_vidN}/*.jpg
- ${REAL_ESTATE_10K}/frames/test/${path_totest_vidN}.txt
```

where `${REAL_ESTATE_10K}/frames/train/video_loc.txt` contains:

```bash
${path_totrain_vid1}
...
${path_totrain_vidN}
```

#### Downloading Our Test Set
We test on a small subset (3.6k) of frames from RealEstate10K. To facilitate replicability, we will share these frames privately with users who have already agreed to terms for downloaded of RealEstate10K and agreed to the terms of [this Google Form](https://docs.google.com/forms/d/e/1FAIpQLScHFr6j-DAi5uXcx2xc0jGt5oQ1CIKqzMTJZ1ywtthUR0YFyw/viewform). 

#### Update options

Update the paths in `./options/options.py` for the dataset being used.

#### Training & Evaluation Note

Some variables containing data in shell scripts have been set to my local directory `/x/cnris/...` as an example. CUDA_VISIBLE_DEVICES is also used to show required GPUs of each script. Change both as needed.

### Training

Training is split into 3 components: 

1. Train VQ-VAE
- `sh scripts/extract_vqvae_dset_realestate.sh`: select subset of images from RealEstate10K to use for training (32k) & validation (8k).
- `sh scripts/train_vqvae_realestate.sh`: train for 150 epochs with batch size 120
- `sh scripts/extract_code_realestate.sh`: with trained vqvae, extract codes for 40k set

2. Train the depth, projection, and refinement module (using frozen VQ-VAE)
- `sh scripts/train_dpr_realestate.sh`: train for 250 epochs with batch size 12 on full RealEstate10K dataset
- `sh scripts/extract_pixcnn_orders_realestate.sh`: with trained depth model, extract orderings used for outpainting on 40k set

3. Train Custom-Order PixelCNN++ (using VQ-VAE embeddings and orderings from depth model)
- `sh scripts/train_lmconv_realestate.sh`: train for 150 epochs with batch size 120

### Evaluation

TIP: Autoregressive sampling is slow - especially if using many samples! Below evaluation code is run on a single GPU; we recommend running multiple times across GPUs with different splits of the evaluation set for faster inference.

#### Pretrained Model and SynSin Baselines
Our pretrained model is available for download [here](https://fouheylab.eecs.umich.edu/~cnris/pixelsynth/modelcheckpoints/realestate/pixelsynth.pth). SynSin - 6X (Our main baseline - SynSin trained on rotations consistent with our model) is available [here](https://fouheylab.eecs.umich.edu/~cnris/pixelsynth/modelcheckpoints/realestate/synsin_6x.pth).
SynSin and several other baselines are available for download at its [Github](https://github.com/facebookresearch/synsin). Place these models in new directory modelcheckpoints/realestate.

#### Evaluating Quality

Evaluating quality evaluates single predicted output given an input image and camera transform. Evaluating consists of two steps:

1. `sh scripts/eval_quality_realestate.sh`: predict output for all images.
2. `python calc_errors_quality.py`: compare outputs to ground truth using FID, Perc Sim and PSNR. Adjust the variables `names`, `base`, `imagedir`, `copy`, and `sampled` as needed

Updated versions of Pytorch, seeding, etc. mean results will closely match paper but not exactly.
Results generated by our model precisely replicating paper results are available [here](https://fouheylab.eecs.umich.edu/~cnris/pixelsynth/our_results/quality/realestate/ours.zip).

#### Evaluating Consistency

Evaluating consistency evaluates the consistency of two predicted outputs given an input image and camera transform. 
The first output is this full rotation and translation, the second is halfway between this and the input. 

We use two methods of evaluating consistency.

1. Using camera transformations involving both rotation and translation via `sh scripts/eval_consistency_realestate.sh`. There is no clear automated metric to evaluate, so we rely on A/B testing to compare consistency across models.
2. Keeping position fixed and applying only rotation. This allows us to use homography to compare consistency of overlapping predicted regions using Perc Sim and PSNR. First, download relevant information to compute homographies ([here](https://fouheylab.eecs.umich.edu/~cnris/pixelsynth/our_results/consistency/realestate/consistency_directions.npy), [here](https://fouheylab.eecs.umich.edu/~cnris/pixelsynth/our_results/consistency/realestate/consistency_masks.zip), and [here](https://fouheylab.eecs.umich.edu/~cnris/pixelsynth/our_results/consistency/realestate/consistency_reference_points.zip)), and put these in `data/` and unzip. Next, run `sh scripts/eval_consistency_homography_realestate.sh`, then `python calc_errors_consistency_homography_realestate.py`, changing variables similar to `calc_errors.py` as needed.

Updated versions of Pytorch, seeding, etc. mean results will closely match paper but not exactly.
Results generated by our model precisely replicating paper results are available [here](https://fouheylab.eecs.umich.edu/~cnris/pixelsynth/our_results/consistency/realestate/ours.zip) for (1) and [here](https://fouheylab.eecs.umich.edu/~cnris/pixelsynth/our_results/consistency/realestate/ours_homography.zip) for homography (2).
