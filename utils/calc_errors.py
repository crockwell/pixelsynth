import numpy as np
from tqdm import tqdm
import os
import cv2
import torchvision.transforms.functional as TF
from PIL import Image
from evaluation.metrics import perceptual_sim, psnr, ssim_metric
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["DEBUG"] = "False"
device = "cuda:0"

#name = 'end_to_end_6x_Res_Perc_Smooth_pretrain'
#names = ['synsin_1x','synsin_6x','end_to_end_6x_Res_Perc_Smooth_pretrain','end_to_end_6x_Res_Perc_Smooth_e2e']
names = ['e2e_final_20-60_vqvae_frozen_ep150_.5', 'e2e_final_20-60_vqvae_frozen_ep100_.5']
#names['synsin_1x','synsin_6x','end_to_end_3x_Percep_KernelSmooth_pretrain','end_to_end_3x_Percep_KernelSmooth_e2e','end_to_end_3x_Percep_KernelSmooth_e2e_.1tmp',
#'end_to_end_6x_vqvae_e2e_ep85_.5tmp','end_to_end_6x_vqvae_e2e_ep85']
#base='/Pool1/users/cnris/synsin/output/mp3d_final_eval/'
#base='/Pool1/users/cnris/synsin/output/realestate_final_eval/'
base='/Pool1/users/cnris/synsin/output/realestate_final_eval_20-60/'
max_img=3600
#if 'realestate' in base:
#    max_img = 3664

COPY=True

# usually dont use these
MAE=False
PERC=False

# find / use oracle predictions if have many
FIND_ORACLE=False
USE_ORACLE=False

# dont use sampled for realestate
SAMPLED=False

# must use OG to use others
OG=True
FID=True
TAILS=True
LPIPS=True
INCEP=True

WRITE_RESULTS=True

def psnr_mask(pred_imgs, key, invis=False):
    mask = pred_imgs["OutputImg"] == pred_imgs["SampledImg"]
    mask = mask.float().min(dim=1, keepdim=True)[0]

    if invis:
        mask = 1 - mask
        
    return psnr(pred_imgs["OutputImg"], pred_imgs[key], mask)


def ssim_mask(pred_imgs, key, invis=False):
    mask = pred_imgs["OutputImg"] == pred_imgs["SampledImg"]
    mask = mask.float().min(dim=1, keepdim=True)[0]

    if invis:
        mask = 1 - mask

    return ssim_metric(pred_imgs["OutputImg"], pred_imgs[key], mask)


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
    "SSIM": lambda pred_imgs, key: ssim_metric(
        pred_imgs["OutputImg"], pred_imgs[key]
    ),
    "SSIM_invis": lambda pred_imgs, key: ssim_mask(pred_imgs, key, True),
    "SSIM_vis": lambda pred_imgs, key: ssim_mask(pred_imgs, key, False),
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
    "SSIM": lambda pred_imgs, key: ssim_metric(
        pred_imgs["OutputImg"], pred_imgs[key]
    ),
    "PercSim": lambda pred_imgs, key: perceptual_sim(
        pred_imgs["OutputImg"], pred_imgs[key], vgg16
    ),
    }


for name in names:
    print('doing:',name)
    input_dir = os.path.join(base,name)
    output_dir = os.path.join(base,name,'all')
    oracle_dir = os.path.join(base,name,'oracle')
    gt_dir = os.path.join(base,'ground_truth','output')
    sampled_dir = os.path.join(base,'ground_truth','sampled_masks')

    try:
        os.makedirs(output_dir)
    except:
        pass

    if WRITE_RESULTS:
        file_to_write = open(os.path.join(input_dir, "our_results_test.txt"),'w')

    if COPY:
        print('copying')
        for i in tqdm(range(max_img)):
            curr_path = os.path.join(input_dir, '{0:04}'.format(i), "output_image_.png") #tgt_image_
            created_path = os.path.join(output_dir, str(i)+'.png') 
            os.system('cp ' + curr_path + ' ' + created_path)

    if MAE:
        mae = np.zeros(max_img)
        print('calculating MAE')
        for i in tqdm(range(max_img)):
            pred_path = os.path.join(output_dir, str(i)+'.png') 
            gt_path = os.path.join(gt_dir, str(i)+'.png') 
            pred_img = cv2.imread(pred_path)
            gt_img = cv2.imread(gt_path)
            diff = abs(pred_img-gt_img)
            mae[i] = np.mean(diff)
        print('using preds',output_dir.split('/')[-1])
        print('MAE mean',np.mean(mae),'std',np.std(mae),'max',np.max(mae),'min',np.min(mae))
        print('MAE % within 10',np.sum(np.where(mae<33,1,0))/max_img,'% within 50',np.sum(np.where(mae<66,1,0))/max_img,'% within 100',np.sum(np.where(mae<100,1,0))/max_img)

    if FIND_ORACLE:
        try:
            os.makedirs(oracle_dir)
        except:
            pass
        print('finding highest percsim for each pair')
        from evaluation.metrics import perceptual_sim
        from models.networks.pretrained_networks import PNet
        device = "cuda:0"
        # Load VGG16 for feature similarity
        vgg16 = PNet().to(device)
        vgg16.eval()
        for i in tqdm(range(max_img)):
            best_num=0
            best_score=9999
            for j in range(50):
                pred_path = os.path.join(input_dir, '{0:04}'.format(i), "output_image_%04d.png"%(j))
                gt_path = os.path.join(gt_dir, str(i)+'.png') 
                pred_img = TF.to_tensor(Image.open(pred_path)).to(device)
                gt_img = TF.to_tensor(Image.open(gt_path)).to(device)
                diff = perceptual_sim(pred_img.unsqueeze_(0),gt_img.unsqueeze_(0),vgg16).cpu().item()
                if diff < best_score:
                    best_score = diff
                    best_num = j
            curr_path = os.path.join(input_dir, '{0:04}'.format(i), "output_image_%04d.png"%(best_num)) #tgt_image_
            created_path = os.path.join(input_dir, '{0:04}'.format(i), "output_image_oracle.png")
            created_path2 = os.path.join(oracle_dir, str(i)+'.png') 
            os.system('cp ' + curr_path + ' ' + created_path)
            os.system('cp ' + curr_path + ' ' + created_path2)
    
    if USE_ORACLE:
        output_dir = oracle_dir

    if PERC:
        from evaluation.metrics import perceptual_sim
        from models.networks.pretrained_networks import PNet
        device = "cuda:0"
        # Load VGG16 for feature similarity
        vgg16 = PNet().to(device)
        vgg16.eval()
        print('calculating PERCSIM')
        ps = np.zeros(max_img)
        for i in tqdm(range(max_img)):
            pred_path = os.path.join(output_dir, str(i)+'.png') 
            gt_path = os.path.join(gt_dir, str(i)+'.png') 
            pred_img = TF.to_tensor(Image.open(pred_path)).to(device)
            gt_img = TF.to_tensor(Image.open(gt_path)).to(device)
            diff = perceptual_sim(pred_img.unsqueeze_(0),gt_img.unsqueeze_(0),vgg16)
            ps[i] = float(diff.cpu()[0])
            #import pdb; pdb.set_trace()
            #if i > 3:
            #    break
        print('using preds',output_dir.split('/')[-1])
        print('PERC SIM mean',np.mean(ps),'std',np.std(ps),'max',np.max(ps),'min',np.min(ps))

    if LPIPS:
        import lpips
        loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        alexs = []
        vggs = []

    all_imgs = []
    if OG:
        from evaluation.metrics import perceptual_sim
        from models.networks.pretrained_networks import PNet
        from torch.utils.data import DataLoader
        import cv2
        import torch
        # Load VGG16 for feature similarity
        vgg16 = PNet().to(device)
        vgg16.eval()
        print('calculating PERCSIM')
        results_all = {}
        for i in tqdm(range(max_img)):
            #try:
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
        
            if LPIPS:
                alexs.append(loss_fn_alex(pred_imgs['PredImg']*2-1, pred_imgs['OutputImg']*2-1).cpu().item())
                vggs.append(loss_fn_vgg(pred_imgs['PredImg']*2-1, pred_imgs['OutputImg']*2-1).cpu().item())
            #except:
            #    print('it failed')
            #    pass
            
            #import pdb; pdb.set_trace()
            #if i > 20:
            #    break
        print('using preds',output_dir.split('/')[-2])
        for metric, result in results_all.items():
            print("%s \t %0.5f \n" % (metric, np.mean(result)))
            file_to_write.write("%s \t %0.5f \n" % (metric, np.mean(result)))
        if LPIPS:
            print('lpips',np.mean(np.array(alexs)), np.mean(np.array(vggs)))
            file_to_write.write("%s \t %0.5f \t %0.5f \n" % ('lpips (alex, vgg)', np.mean(np.array(alexs)), np.mean(np.array(vggs))))
        if FID:
            os.system('python -m pytorch_fid '+output_dir+' '+gt_dir)
        if TAILS:
            print("%s \t %0.5f" % ('PSNR > 20', np.mean(np.where(np.array(results_all['PSNR'])>20,1,0))))
            print("%s \t %0.5f" % ('PercSim < 2.3', np.mean(np.where(np.array(results_all['PercSim'])<2.3,1,0))))
            print("%s \t %0.5f" % ('SSIM > .8', np.mean(np.where(np.array(results_all['SSIM'])>.8,1,0))))
            file_to_write.write("%s \t %0.5f \n" % ('PSNR > 20', np.mean(np.where(np.array(results_all['PSNR'])>20,1,0))))
            file_to_write.write("%s \t %0.5f \n" % ('PercSim < 2.3', np.mean(np.where(np.array(results_all['PercSim'])<2.3,1,0))))
            file_to_write.write("%s \t %0.5f \n" % ('PSNR > .8', np.mean(np.where(np.array(results_all['SSIM'])>.8,1,0))))
            #print(np.where(np.array(results_all['PSNR'])>20,1,0).nonzero())
            #print(np.where(np.array(results_all['PercSim'])<2.3,1,0).nonzero())
            #print(np.where(np.array(results_all['SSIM'])>.8,1,0).nonzero())
        if INCEP:
            from inception_score import get_inception_score
            sc=get_inception_score(all_imgs)
            print('inception score mean',sc[0],'std',sc[1])
            file_to_write.write("%s \t %s \t %s \n" % ('inception score mean, sd', str(sc[0]), str(sc[1])))
        file_to_write.close()
        #print('PERC SIM mean',np.mean(ps),'std',np.std(ps),'max',np.max(ps),'min',np.min(ps))