# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import time

import torch
import torch.nn as nn
import torchvision
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader

from models.base_model import BaseModel
from models.networks.sync_batchnorm import convert_model
from options.options import get_dataset, get_model
from options.train_options import (
    ArgumentParser,
    get_model_path,
    get_timestamp,
)
import pickle as pkl
from tqdm import tqdm
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

torch.backends.cudnn.benchmark = True

def run(data_loader, model, opts):
    trainval = opts.dataset_folder.split('/')[-1]
    print("Loaded "+trainval+" dataset ...", flush=True)
    with torch.no_grad():
        gen_order = {}

        dset_size = int(32000 / opts.batch_size)
        all_cameras = {}
        if trainval == 'val':
            dset_size = int(8000 / opts.batch_size)

        losses = {}
        iter_data_loader = iter(data_loader)

        for i in tqdm(range(dset_size)):
            batch = next(iter_data_loader)
            _, output_images, _ = model(
                    batch, isval=True, return_batch=True
            )

            gens = output_images['gen_order'].cpu().data

            for j in range(opts.batch_size):
                gen_order[i*opts.batch_size+j] = gens[j]
        
        with open('data/%s_%s_gen_order.pkl' % (opts.dataset.split('_')[-1], trainval), 'wb') as f:
            pkl.dump(gen_order, f)

    return {l: losses[l] / float(iteration) for l in losses.keys()}

if __name__ == "__main__":
    torch.cuda.empty_cache()
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    test_ops, _ = ArgumentParser().parse()

    

    # Load model to be tested
    MODEL_PATH = test_ops.old_model
    BATCH_SIZE = test_ops.batch_size

    
    opts = torch.load(MODEL_PATH)["opts"]
    opts.config = 'habitat-lab/configs/tasks/pointnav_rgbd.yaml'
    opts.dataset_folder = test_ops.dataset_folder
    opts.dataset = test_ops.dataset 
    opts.model_setting = test_ops.model_setting
    opts.batch_size = test_ops.batch_size

    Dataset = get_dataset(opts)
    data = Dataset("test", opts)
    model = get_model(opts)
    dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=BATCH_SIZE,
        )

    torch_devices = [0]
    device = "cuda:" + str(torch_devices[0])

    
    if "sync" in opts.norm_G:
        model = convert_model(model)
        model = nn.DataParallel(model, torch_devices).to(device)
    else:
        model = nn.DataParallel(model, torch_devices).to(device)
    

    #  Load the original model to be tested
    model = BaseModel(model, opts)
    model.eval()

    # Allow for different image sizes
    state_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in torch.load(MODEL_PATH)["state_dict"].items()
        if not ("xyzs" in k) and not ("ones" in k)
    }
    state_dict.update(pretrained_dict)

    model.load_state_dict(state_dict)
    
    print('loaded old model')

    run(dataloader, model, opts)
