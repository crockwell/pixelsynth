# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch.utils.data as data
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import pickle as pkl

from utils.geometry import get_deltas
import os

class CustomTest(data.Dataset):
    """ Dataset for loading the custom data (for testing)
    """

    def __init__(
        self, dataset, opts=None
    ):
        # get pkl file & image locations

        with open(os.path.join(opts.dataset_folder, 'cameras.pkl'),'rb') as f:
            self.cameras = pkl.load(f)

        # used for consistency eval only
        self.consistency_directions = np.load('data/consistency_directions.npy')

        self.images = {}
        for set in ['input','output']:
            dir = os.path.join(opts.dataset_folder, set)
            unsorted_paths = {}
            for root, dnames, fnames in sorted(os.walk(dir)):
                for fname in fnames:
                    if fname.endswith('.png'):
                        path = os.path.join(root, fname)
                        unsorted_paths[int(path.split('/')[-1][:-4])] = path
                        #if set not in self.images:
                        #    self.images[set] = [path]
                        #else:
                        #    self.images[set].append(path)
            sorted_paths = []
            for path in sorted(unsorted_paths):
                sorted_paths.append(unsorted_paths[path])
            self.images[set] = sorted_paths

        self.input_transform = Compose(
            [
                Resize((opts.W, opts.W)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.opts = opts

    def __len__(self):
        return len(self.images['input'])

    def __getitem__(self, image_number):        
        image = [Image.open(self.images['input'][image_number]), Image.open(self.images['output'][image_number])]
        img = [self.input_transform(image[0]), self.input_transform(image[1])]
        direction = torch.tensor(self.consistency_directions[image_number])
        newcameras = []
        for i in range(2):
            camera = {}
            for key in self.cameras[image_number][i].keys():
                if key not in ['translation','angle','is_big_change','frame_diff','vid_names']:
                    camera[key] = self.cameras[image_number][i][key][0]
            newcameras.append(camera)

        return {"images": img, "cameras": newcameras, 'direction': direction}


class Custom(data.Dataset):
    """ Dataset for loading the custom data (used after has been generated for vqvae/lmconv training)
    """

    def __init__(
        self, dataset, opts=None
    ):
        # get pkl file & image locations

        with open(os.path.join(opts.dataset_folder, 'cameras.pkl'),'rb') as f:
            self.cameras = pkl.load(f)

        dir = os.path.join(opts.dataset_folder, 'rgb')
        self.images = []
        for root, dnames, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname.endswith('.png'):
                    path = os.path.join(root, fname)                
                    self.images.append(path)
            

        self.input_transform = Compose(
            [
                Resize((opts.W, opts.W)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        #import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, image_number):
        
        image = Image.open(self.images[image_number])
        img = [self.input_transform(image)]
        newcameras = []
        for i in range(2):
            camera = {}
            for key in self.cameras[image_number][i].keys():
                if key not in ['translation','angle','is_big_change','frame_diff','vid_names']:
                    camera[key] = self.cameras[image_number][i][key][0]
            newcameras.append(camera)

        return {"images": img, "cameras": newcameras}
