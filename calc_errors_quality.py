import numpy as np
from tqdm import tqdm
import os
import cv2
import torchvision.transforms.functional as TF
from PIL import Image
from evaluation.metrics import perceptual_sim, psnr, ssim_metric
from models.networks.pretrained_networks import PNet
from torch.utils.data import DataLoader
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["DEBUG"] = "False"
device = "cuda:0"

## input variables
names = ['ours']
base = 'output/images/realestate/'
imagedir = '/x/cnris/realestate_eval/'
# if using new predictions, will need to copy images first time script is run 
COPY=False
# sampled not available for realestate
SAMPLED=False

max_img=3600
## end input variables

def psnr_mask(pred_imgs, key, invis=False):
    mask = pred_imgs["OutputImg"] == pred_imgs["SampledImg"]
    mask = mask.float().min(dim=1, keepdim=True)[0]

    if invis:
        mask = 1 - mask
        
    return psnr(pred_imgs["OutputImg"], pred_imgs[key], mask)


def perceptual_sim_mask(pred_imgs, key, vgg16, invis=False):
    mask = pred_imgs["OutputImg"] == pred_imgs["SampledImg"]
    mask = mask.float().min(dim=1, keepdim=True)[0]

    if invis:
        mask = 1 - mask

    return perceptual_sim(
        pred_imgs["OutputImg"] * mask, pred_imgs[key] * mask, vgg16
    )

if SAMPLED:
    METRICS = {
    "PSNR": lambda pred_imgs, key: psnr(
        pred_imgs["OutputImg"], pred_imgs[key]
    ).clamp(max=100),
    "PSNR_invis": lambda pred_imgs, key: psnr_mask(
        pred_imgs, key, True
    ).clamp(max=100),
    "PSNR_vis": lambda pred_imgs, key: psnr_mask(
        pred_imgs, key, False
    ).clamp(max=100),
    "PercSim": lambda pred_imgs, key: perceptual_sim(
        pred_imgs["OutputImg"], pred_imgs[key], vgg16
    ),
    "PercSim_invis": lambda pred_imgs, key: perceptual_sim_mask(
        pred_imgs, key, vgg16, True
    ),
    "PercSim_vis": lambda pred_imgs, key: perceptual_sim_mask(
        pred_imgs, key, vgg16, False
    ),
    }
else:
    METRICS = {
    "PSNR": lambda pred_imgs, key: psnr(
        pred_imgs["OutputImg"], pred_imgs[key]
    ).clamp(max=100),
    "PercSim": lambda pred_imgs, key: perceptual_sim(
        pred_imgs["OutputImg"], pred_imgs[key], vgg16
    ),
    }


for name in names:
    print('doing:',name)
    input_dir = os.path.join(base,name)
    output_dir = os.path.join(base,name,'all')
    gt_dir = os.path.join(imagedir,'output')
    if SAMPLED:
        sampled_dir = os.path.join(imagedir,'sampled_masks')
        try:
            os.makedirs(sampled_dir)
        except:
            pass

    try:
        os.makedirs(output_dir)
    except:
        pass

    if COPY:
        print('copying')
        for i in tqdm(range(max_img)):
            curr_path = os.path.join(input_dir, '{0:04}'.format(i), "output_image_.png") #tgt_image_
            created_path = os.path.join(output_dir, str(i)+'.png') 
            os.system('cp ' + curr_path + ' ' + created_path)

    all_imgs = []
    # Load VGG16 for feature similarity
    vgg16 = PNet().to(device)
    vgg16.eval()
    print('calculating PSNR and PercSim')
    results_all = {}
    for i in tqdm(range(max_img)):
        pred_path = os.path.join(output_dir, str(i)+'.png') 
        gt_path = os.path.join(gt_dir, str(i)+'.png') 
        pred_imgs = {}
        if SAMPLED:
            sampled_path = os.path.join(sampled_dir, str(i)+'.png') 
            pred_imgs['SampledImg'] = TF.to_tensor(Image.open(sampled_path)).to(device).unsqueeze(0)
        predimg = Image.open(pred_path)
        pred_imgs['PredImg'] = TF.to_tensor(predimg).to(device).unsqueeze(0)
        pred_imgs['OutputImg'] = TF.to_tensor(Image.open(gt_path)).to(device).unsqueeze(0)
        
        all_imgs.append(np.array(predimg))
        
        for metric, func in METRICS.items():
            t_results = func(pred_imgs, "PredImg") 
            
            if not (metric in results_all.keys()):
                results_all[metric] = [t_results.sum().cpu().item()]
            else:
                results_all[metric].append(t_results.sum().cpu().item())
    
    print('using preds',output_dir.split('/')[-2])
    for metric, result in results_all.items():
        print("%s \t %0.5f \n" % (metric, np.mean(result)))
    os.system('python -m pytorch_fid '+output_dir+' '+gt_dir)
