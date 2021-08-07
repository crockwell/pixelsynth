# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os

import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader
import torchvision

from options.options import get_dataset
from options.train_options import ArgumentParser
from tqdm import tqdm
import pickle as pkl

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

torch.backends.cudnn.benchmark = True


def run(dset, trainval, opts):
    print("Loading "+trainval+" dataset ....", flush=True)
    
    data_loader = DataLoader(
        dataset=dset,
        num_workers=opts.num_workers,
        batch_size=opts.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    print("Loaded "+trainval+" dataset ...", flush=True)

    if not os.path.exists(opts.result_folder + "/%s/rgb/" % (trainval)):
        os.makedirs(opts.result_folder + "/%s/rgb/" % (trainval))

    iter_data_loader = iter(data_loader)

    dset_size = int(32000 / opts.batch_size)
    all_cameras = {}
    if trainval == 'val':
        dset_size = int(8000 / opts.batch_size)
    pbar = tqdm(total=dset_size)
    i = 0
    
    while i < dset_size:
        try:
            # sometimes realestate batch fails from missing camera params
            batch = next(iter_data_loader)
        except:
            # try again with next batch
            continue
        # save batch of images
        images = batch['images'][-1]
        if opts.dataset == 'realestate':
            images = images *.5+.5
        for j in range(opts.batch_size):
            cams = []
            for k in range(2):
                cam = {}
                for key in batch['cameras'][k].keys():
                    cam[key] = batch['cameras'][k][key][j:j+1]
                cams.append(cam)
            all_cameras[int(i*opts.batch_size+j)] = cams

            torchvision.utils.save_image(
                images[j],
                opts.result_folder
                + "/%s/rgb/%d.png" % (trainval, int(i*opts.batch_size+j)),
            )
        i += 1
        pbar.update(1)
    pbar.close()

    with open(opts.result_folder + "/%s/cameras.pkl" % (trainval), 'wb') as f:
        pkl.dump(all_cameras, f)

    print("Finished selecting "+trainval+" dataset ....", flush=True)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    opts, _ = ArgumentParser().parse()

    opts.config = 'habitat-lab/configs/tasks/pointnav_rgbd.yaml'

    Dataset = get_dataset(opts)

    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
    print(torch_devices)
    device = "cuda:" + str(torch_devices[0])

    dset = Dataset('train', opts)
    run(dset, 'train', opts)
    dset.toval(
        epoch=0
    )
    run(dset, 'val', opts)
