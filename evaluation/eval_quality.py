import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import transforms as trn
from PIL import Image
from torch.utils.data import DataLoader

import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from models.base_model import BaseModel
from models.networks.sync_batchnorm import convert_model
from options.options import get_dataset, get_model
from options.test_options import ArgumentParser

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["DEBUG"] = "False"
from utils.geometry import get_deltas
from tqdm import tqdm
from utils.opts_helper import opts_helper

if __name__ == "__main__":
    test_ops, _ = ArgumentParser().parse()

    opts = opts_helper(test_ops)
    model = get_model(opts)

    opts.render_ids = test_ops.render_ids
    opts.gpu_ids = test_ops.gpu_ids

    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
    print(torch_devices)
    device = "cuda:" + str(torch_devices[0])

    if "sync" in opts.norm_G:
        model = convert_model(model)
        model = nn.DataParallel(model, torch_devices).to(device)
    else: 
        model = nn.DataParallel(model, torch_devices).to(device)

    #  Load the original model to be tested
    model_to_test = BaseModel(model, opts)
    model_to_test.eval()

    # Allow for different image sizes
    state_dict = model_to_test.state_dict()
    pretrained_dict = {
        k: v
        for k, v in torch.load(test_ops.old_model)["state_dict"].items()
        if not ("xyzs" in k) and not ("ones" in k)
    }
    state_dict.update(pretrained_dict)

    model_to_test.load_state_dict(state_dict,strict=False)

    print(opts)

    print("Loaded models...")

    if test_ops.load_vqvae:
        print('loading vqvae')
        tmp2 = torch.load(test_ops.vqvae_path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in tmp2.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model_to_test.model.module.vqvae.load_state_dict(new_state_dict)
    else:
        print('not loading vqvae')
    if test_ops.load_autoregressive:
        print('loading autoregressive')
        ar_model = torch.load(test_ops.autoregressive)
        model_to_test.model.module.outpaint2.load_state_dict(ar_model['model_state_dict'], strict=False)
    else:
        print('not loading autoregressive')

    if not opts.no_outpainting:
        arch = 'resnet18'
        # load the pre-trained weights for classifier
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model_to_test.model.module.classifier.load_state_dict(state_dict)

    model_to_test.eval()

    Dataset = get_dataset(opts)
    data = Dataset("test", opts)
    dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=test_ops.batch_size,
        )
    iter_data_loader = iter(dataloader)

    for i in tqdm(range(data.__len__())):
        #if i < 2700 or i > 3600:
        #    batch = next(iter_data_loader)
        #    continue
        with torch.no_grad():
            batch = next(iter_data_loader)
            _, pred_imgs, _ = model_to_test(
                batch, isval=True, return_batch=True
            )

        if not os.path.exists(
            test_ops.result_folder
                + "/%04d/" % (i)
        ):
            os.makedirs(
                test_ops.result_folder
                + "/%04d/" % (i)
            )

        torchvision.utils.save_image(
            pred_imgs["OutputImg"],
            test_ops.result_folder
            + "/%04d/tgt_image_.png" % (i)
        )

        torchvision.utils.save_image(
            pred_imgs["InputImg"],
            test_ops.result_folder
            + "/%04d/input_image_.png" % (i),
        )

        if pred_imgs["FeaturesImg"].shape[1] == 3:
            torchvision.utils.save_image(
                pred_imgs["FeaturesImg"],
                test_ops.result_folder
                + "/%04d/input_fs_image_.png" % (i),
            )

        torchvision.utils.save_image(
            pred_imgs["PredImg"],
            test_ops.result_folder
            + "/%04d/output_image_.png" % (i),
        )