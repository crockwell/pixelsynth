import numpy as np
from tqdm import tqdm
import os
import cv2
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from evaluation.metrics import perceptual_sim, psnr, ssim_metric
from evaluation.metrics import perceptual_sim
from models.networks.pretrained_networks import PNet
from torch.utils.data import DataLoader
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["DEBUG"] = "False"
device = "cuda:0"

names = ['ours']
base='output/consistency_homography/realestate/'
imgset = np.arange(3600)
these_directions = np.load('data/consistency_directions.npy')

mapping = ['R','L','U','D', 'UL', 'UR', 'DR', 'DL']

def perceptual_sim_mask(warped_img, masked_img, vgg16):
    warped_img = TF.to_tensor(warped_img).to(device).unsqueeze(0)/255
    masked_img = TF.to_tensor(masked_img).to(device).unsqueeze(0)/255

    return perceptual_sim(
        warped_img, masked_img, vgg16
    )

def psnr_mask(warped_img, masked_img, mask):
    warped_img = TF.to_tensor(warped_img).unsqueeze(0)/255
    masked_img = TF.to_tensor(masked_img).unsqueeze(0)/255
    mask = TF.to_tensor(mask).unsqueeze(0)

    return psnr(
        warped_img, masked_img, mask
    )

METRICS = {
"PercSim_vis": lambda warped_img, masked_img, mask: perceptual_sim_mask(
    warped_img, masked_img, vgg16
), 
"PSNR_vis": lambda warped_img, masked_img, mask: psnr_mask(
        warped_img, masked_img, mask
).clamp(max=100),
}

for name in names:
    print('doing:',name)
    input_dir = os.path.join(base,name)
    mask_dir = os.path.join('data/consistency_masks')
    points_dir = os.path.join('data','consistency_reference_points')

    all_imgs = []
    # Load VGG16 for feature similarity
    vgg16 = PNet().to(device)
    vgg16.eval()
    print('calculating PERCSIM')
    results_all = {}
    for i, imgnum in tqdm(enumerate(imgset)):
        path1 = os.path.join(input_dir, '%04d' % (imgnum), 'output_image_%s_0001.png' % mapping[these_directions[i]])#i[0]) 
        path2 = os.path.join(input_dir, '%04d' % (imgnum), 'output_image_%s_0002.png' % mapping[these_directions[i]])#i[1]) 
        pred_imgs = {}
        sampled_path1 = os.path.join(mask_dir, '%04d' % (imgnum), 'mask1.png') 
        sampled_path2 = os.path.join(mask_dir, '%04d' % (imgnum), 'mask2.png')
        pred_imgs['mask1'] = TF.to_tensor(ImageOps.grayscale(Image.open(sampled_path1))).to(device).unsqueeze(0)
        pred_imgs['mask2'] = TF.to_tensor(ImageOps.grayscale(Image.open(sampled_path2))).to(device).unsqueeze(0)
        pred_img1 = np.array(Image.open(path1))[:,:,::-1]
        pred_img2 = np.array(Image.open(path2))[:,:,::-1]
        pred_imgs['Img1'] = TF.to_tensor(Image.open(path1)).to(device).unsqueeze(0)
        pred_imgs['Img2'] = TF.to_tensor(Image.open(path2)).to(device).unsqueeze(0)

        mask1 = pred_imgs['mask1'][0].reshape((256,256,1)).cpu().numpy()
        mask2 = pred_imgs['mask2'][0].reshape((256,256,1)).cpu().numpy()
        outmask = np.swapaxes(np.swapaxes((pred_imgs['mask1']*pred_imgs['Img1'])[0].cpu().numpy(), 0, 2),0,1)[:,:,::-1]*255
        outmask2 = np.swapaxes(np.swapaxes((pred_imgs['mask2']*pred_imgs['Img2'])[0].cpu().numpy(), 0, 2),0,1)[:,:,::-1]*255
        try1 = np.swapaxes(np.swapaxes((pred_imgs['Img1'])[0].cpu().numpy(), 0, 2),0,1)[:,:,::-1]*255
        try2 = np.swapaxes(np.swapaxes((pred_imgs['Img2'])[0].cpu().numpy(), 0, 2),0,1)[:,:,::-1]*255

        pts_src = (np.load(os.path.join(points_dir,'reproj1_%d.npy' % (imgnum)))*.5+.5)*255
        pts_dst = (np.load(os.path.join(points_dir,'reproj2_%d.npy' % (imgnum)))*.5+.5)*255
        pts_src[:,0] = 255-pts_src[:,0]
        pts_dst[:,0] = 255-pts_dst[:,0]
                    
        h, status = cv2.findHomography(pts_src[:,:2], pts_dst[:,:2])
        h2, status = cv2.findHomography(pts_dst[:,:2], pts_src[:,:2])
        im_out2 = cv2.warpPerspective(try1, h, (outmask2.shape[1],outmask2.shape[0]))
        im_out = cv2.warpPerspective(try2, h2, (outmask2.shape[1],outmask2.shape[0]))

        cv2.imwrite(os.path.join(input_dir, '%04d' % (imgnum),'1_masked.png'), pred_img1*(mask1))
        cv2.imwrite(os.path.join(input_dir, '%04d' % (imgnum),'2_masked.png'), pred_img2*(mask2))
        cv2.imwrite(os.path.join(input_dir, '%04d' % (imgnum),'2_warped_1_unmasked.png'), im_out)
        cv2.imwrite(os.path.join(input_dir, '%04d' % (imgnum),'1_warped_2_unmasked.png'), im_out2)
        cv2.imwrite(os.path.join(input_dir, '%04d' % (imgnum),'2_warped_1_masked.png'), im_out*(mask1))
        cv2.imwrite(os.path.join(input_dir, '%04d' % (imgnum),'1_warped_2_masked.png'), im_out2*(mask2))
    
        for metric, func in METRICS.items():
            t_results = func(im_out*mask1, outmask, mask1) 
            t_results2 = func(im_out2*mask2, outmask2, mask2) 
            if not (metric in results_all.keys()):
                results_all[metric] = [(t_results.sum().cpu().item()+t_results2.sum().cpu().item())*.5]
            else:
                results_all[metric].append((t_results.sum().cpu().item()+t_results2.sum().cpu().item())*.5)
        
    print('using preds',input_dir.split('/')[-2])
    for metric, result in results_all.items():
        print("%s \t %0.5f \n" % (metric, np.mean(result)))