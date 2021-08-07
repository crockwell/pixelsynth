import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import transforms as trn
from PIL import Image
from torch.utils.data import DataLoader

from models.base_model import BaseModel
from models.networks.sync_batchnorm import convert_model
from options.options import get_dataset, get_model
from options.test_options import ArgumentParser

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["DEBUG"] = "False"
from utils.geometry import get_deltas
from tqdm import tqdm
from utils.opts_helper import opts_helper

def process_demo_data(opts=None):
    input_transform = trn.Compose(
        [
            trn.Resize((opts.W, opts.W)),
            trn.ToTensor(),
            trn.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    offset = np.array(
        [[2, 0, -1], [0, -2, 1], [0, 0, -1]],  # Flip ys to match habitat
        dtype=np.float32,
    )  # Make z negative to match habitat (which assumes a negative z)

    K = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    invK = np.linalg.inv(K)

    rgbs = []
    cameras = []
    image = Image.open(os.path.join('demo',opts.demo_img_name))
    input_shape = np.array(image).shape
    ratio = input_shape[1] / input_shape[0]
    rgbs += [input_transform(image).unsqueeze(0)]

    # The reference camera position can just be the identity
    extrinsics = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    # Accurate intrinsics are only important if we are trying to match a ground
    # truth output. Here we just give intrinsics for the input image with the
    # principal point in the center.
    intrinsics = np.array([1.0, 1.0 * ratio, 0.5, 0.5])

    origK = np.array(
        [
            [intrinsics[0], 0, intrinsics[2]],
            [0, intrinsics[1], intrinsics[3]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    Ktmp = np.matmul(offset, origK)

    origP = extrinsics
    P = np.matmul(Ktmp, origP)  # Merge these together to match habitat
    P = np.vstack((P, np.zeros((1, 4), dtype=np.float32))).astype(
        np.float32
    )
    P[3, 3] = 1

    Pinv = np.linalg.inv(P)

    cameras += [
        {
            "P": torch.tensor(P).unsqueeze(0),
            "Pinv": torch.tensor(Pinv).unsqueeze(0),
            "OrigP": torch.tensor(origP).unsqueeze(0),
            "K": torch.tensor(K).unsqueeze(0),
            "Kinv": torch.tensor(invK).unsqueeze(0),
        }
    ]

    return {"images": rgbs, "cameras": cameras}

def save_scene(pred_imgs, test_ops):
    if not os.path.exists(
        test_ops.result_folder
        + "/scene/"
    ):
        os.makedirs(
            test_ops.result_folder
            + "/scene/"
        )

    for direction in test_ops.directions:
        if direction in ['S','C']:
            continue
            
        num_split = test_ops.num_split
        if direction in ['U','D', 'UL', 'UR', 'DR', 'DL']:
            num_split = max(int(test_ops.num_split/2), 1)

        for i in range(1,num_split+1): 
            torchvision.utils.save_image(
                pred_imgs["PredImg_"+direction+'_'+str(i)],
                test_ops.result_folder
                + "/scene/output_image_%s_%04d.png" % (direction, i),
            )

def save_video(pred_imgs, test_ops):
    if not os.path.exists(
        test_ops.result_folder
        + "/video/"
    ):
        os.makedirs(
            test_ops.result_folder
            + "/video/"
        )
    video_ct = 0
    directions = ['R','L', 'C', 'C', 'S', 'S']
    torchvision.utils.save_image(
        pred_imgs["PredImg_"+'R'+'_'+str(0)],
        test_ops.result_folder
        + "/video/%d.png" % (video_ct),
    )

    video_ct += 1

    for direction in directions:
        num_split = test_ops.num_split
        if direction in ['S','C']:
            num_split = test_ops.num_split * 2

        for i in range(1,num_split): 
            torchvision.utils.save_image(
                pred_imgs["PredImg_"+direction+'_'+str(i)],
                test_ops.result_folder
                + "/video/%d.png" % (video_ct),
            )
            video_ct += 1

        if direction not in ['S', 'C']:
            for i in range(num_split-1,-1,-1): 
                torchvision.utils.save_image(
                    pred_imgs["PredImg_"+direction+'_'+str(i)],
                    test_ops.result_folder
                    + "/video/%d.png" % (video_ct),
                )
                video_ct += 1

def save_img(pred_imgs, test_ops):
    torchvision.utils.save_image(
        pred_imgs["PredImg"],
        test_ops.result_folder
        + "/output_image_%s_%d.png" % (test_ops.direction,test_ops.rotation),
    )

    if pred_imgs["FeaturesImg"].shape[1] == 3:
        torchvision.utils.save_image(
            pred_imgs["FeaturesImg"],
            test_ops.result_folder
            + "/input_fs_image_%s_%d.png" % (test_ops.direction,test_ops.rotation),
        )

if __name__ == "__main__":
    test_ops, _ = ArgumentParser().parse()

    opts = opts_helper(test_ops)
    
    model = get_model(opts)

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

    arch = 'resnet18'
    import torchvision.models as models
    # load the pre-trained weights for classifier
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model_to_test.model.module.classifier.load_state_dict(state_dict)

    model_to_test.eval()

    batch = process_demo_data(opts)
    with torch.no_grad():
        _, pred_imgs, _ = model_to_test(
            batch, isval=True, return_batch=True
        )

        if not os.path.exists(
            test_ops.result_folder
        ):
            os.makedirs(
                test_ops.result_folder
            )

        torchvision.utils.save_image(
            pred_imgs["InputImg"],
            test_ops.result_folder
            + "/input_image_.png"
        )

        if opts.model_setting == 'gen_scene':
            save_scene(pred_imgs, test_ops)
            save_video(pred_imgs, test_ops)
        else:
            save_img(pred_imgs, test_ops)
        