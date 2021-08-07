# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from utils.geometry import get_deltas
import os

class RealEstate10K(data.Dataset):
    """ Dataset for loading the RealEstate10K. In this case, images are randomly 
    chosen within a video subject to certain constraints: e.g. they should 
    be within a number of frames but the angle and translation should
    vary as much as possible.
    """

    def __init__(
        self, dataset, opts=None, num_views=2, seed=0, vectorize=False
    ):
        # Now go through the dataset

        self.imageset = np.loadtxt(
            opts.train_data_path + "/frames/%s/video_loc.txt" % "train",
            dtype=np.str,
        )
        self.base_file = opts.train_data_path

        self.dataset = "train"
        self.isTrain = True

        if dataset == "train":
            self.imageset = self.imageset[0 : int(0.8 * self.imageset.shape[0])]
        elif dataset == "val":
            self.isTrain = False
            self.imageset = self.imageset[int(0.8 * self.imageset.shape[0]) :]
        elif dataset == "test":
            self.isTrain = False
            self.dataset = dataset
            self.imageset = np.loadtxt(
                opts.test_data_path + "/frames/%s/video_loc.txt" % "test",
                dtype=np.str,
            )
            self.base_file = opts.test_data_path

        self.rng = np.random.RandomState(seed)
        

        self.num_views = num_views

        self.input_transform = Compose(
            [
                Resize((opts.W, opts.W)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.offset = np.array(
            [[2, 0, -1], [0, -2, 1], [0, 0, -1]],  # Flip ys to match habitat
            dtype=np.float32,
        )  # Make z negative to match habitat (which assumes a negative z)

        self.K = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.invK = np.linalg.inv(self.K)

        #self.ANGLE_THRESH = 5
        #self.TRANS_THRESH = 0.15

        # TODO: how do we track angle increases during training?
        self.opts = opts
        #self.ANGLE_THRESH = opts.max_rotation
        #self.TRANS_THRESH = 99999
        if self.isTrain:
            print('train angle threshold', self.opts.max_rotation//2)
        else:
            print('val angle threshold', self.opts.val_rotation//2)

    def __len__(self):
        return 2 ** 31

    def __getitem_simple__(self, index):
        index = self.rng.randint(self.imageset.shape[0])
        # index = index % self.imageset.shape[0]
        # Load text file containing frame information
        frames = np.loadtxt(
            self.base_file
            + "/frames/%s/%s.txt" % (self.dataset, self.imageset[index])
        )

        image_index = self.rng.choice(frames.shape[0], size=(1,))[0]

        rgbs = []
        cameras = []
        for i in range(0, self.num_views):
            t_index = max(
                min(
                    image_index + self.rng.randint(16) - 8, frames.shape[0] - 1
                ),
                0,
            )

            image = Image.open(
                self.base_file
                + "/frames/%s/%s/" % (self.dataset, self.imageset[index])
                + str(int(frames[t_index, 0]))
                + ".png"
            )
            rgbs += [self.input_transform(image)]

            intrinsics = frames[t_index, 1:7]
            extrinsics = frames[t_index, 7:]

            origK = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            K = np.matmul(self.offset, origK)

            origP = extrinsics.reshape(3, 4)
            P = np.matmul(K, origP)  # Merge these together to match habitat
            P = np.vstack((P, np.zeros((1, 4), dtype=np.float32))).astype(
                np.float32
            )
            P[3, 3] = 1

            Pinv = np.linalg.inv(P)

            cameras += [
                {
                    "P": P,
                    "OrigP": origP,
                    "Pinv": Pinv,
                    "K": self.K,
                    "Kinv": self.invK,
                }
            ]

        return {"images": rgbs, "cameras": cameras}

    def __getitem__(self, image_number):
        if self.isTrain:
            angle_threshold = self.opts.max_rotation//2
        else:
            angle_threshold = self.opts.val_rotation//2

        satisfied_large_angle = False
        while not satisfied_large_angle:
            index = self.rng.randint(self.imageset.shape[0])
            # index = index % self.imageset.shape[0]
            # Load text file containing frame information

            # My data has header row (youtube file name), which we skip
            # must also check have at least 2 frames; i.e. be 2 dimensional
            satisfied_2framereq = False
            while not satisfied_2framereq:
                frames = np.loadtxt(
                    self.base_file
                    + "/frames/%s/%s.txt" % (self.dataset, self.imageset[index]), skiprows=1
                )
                if True: #len(frames.shape) > 1:
                    satisfied_2framereq = True
                else:
                    index = self.rng.randint(self.imageset.shape[0])

            image_index = self.rng.choice(frames.shape[0], size=(1,))[0]
            

            # Chose 15 images within 30 frames of the iniital one
            #image_indices = self.rng.randint(180, size=(45,)) - 90 + image_index
            #image_indices = self.rng.randint(80, size=(30,)) - 40 + image_index
            #image_indices = self.rng.randint(60, size=(15,)) - 30 + image_index
            #image_indices = np.minimum(
            #    np.maximum(image_indices, 0), frames.shape[0] - 1
            #)

            # instead, we use second image index from the entire video
            # we'll have to check in the future how long the videos are
            image_indices = self.rng.randint(frames.shape[0]-1, size=(frames.shape[0]//2,))

            # Look at the change in angle and choose a hard one
            angles = []
            translations = []
            for viewpoint in range(0, image_indices.shape[0]):
                orig_viewpoint = frames[image_index, 7:].reshape(3, 4)
                new_viewpoint = frames[image_indices[viewpoint], 7:].reshape(3, 4)
                dang, dtrans = get_deltas(orig_viewpoint, new_viewpoint)

                angles += [dang]
                translations += [dtrans]

            angles = np.array(angles)
            translations = np.array(translations)

            #mask = image_indices[
            #    (angles > self.ANGLE_THRESH) | (translations > self.TRANS_THRESH)
            #]
            mask = image_indices[
                np.logical_and(np.logical_and((angles > angle_threshold), (translations < 1)), (angles < 60))
            ]
            
            if len(mask) > 5:
                satisfied_large_angle = True

        #import pdb 
        #pdb.set_trace()

        rgbs = []
        cameras = []
        final_translation = []
        final_angle = []
        frame_diff = []
        for i in range(0, self.num_views):
            if i == 0:
                t_index = image_index
            elif mask.shape[0] > 5:
                # Choose a harder angle change
                numba = self.rng.randint(mask.shape[0])
                t_index = mask[numba]
                masked_angles = angles[(angles > angle_threshold)]
                masked_translations = translations[(angles > angle_threshold)]
                final_angle = masked_angles[numba]
                final_translation = masked_translations[numba]
                frame_diff = abs(numba-image_index)
            else:
                numba = self.rng.randint(image_indices.shape[0])
                t_index = image_indices[
                    numba
                ]
                final_angle = angles[numba]
                final_translation = translations[numba]
                frame_diff = abs(numba-image_index)
            
            #try:
            image = Image.open(
                self.base_file
                + "/frames/%s/%s/" % (self.dataset, self.imageset[index])
            + str(int(frames[t_index, 0]))
                + ".jpg"
            )
            #except:
            #    #import pdb 
            #    #pdb.set_trace()
            rgbs += [self.input_transform(image)]

            intrinsics = frames[t_index, 1:7]
            extrinsics = frames[t_index, 7:]

            origK = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            K = np.matmul(self.offset, origK)

            origP = extrinsics.reshape(3, 4)
            P = np.matmul(K, origP)  # Merge these together to match habitat
            P = np.vstack((P, np.zeros((1, 4), dtype=np.float32))).astype(
                np.float32
            )
            P[3, 3] = 1

            Pinv = np.linalg.inv(P)

            cameras += [
                {
                    "P": P,
                    "Pinv": Pinv,
                    "OrigP": origP,
                    "K": self.K,
                    "Kinv": self.invK,
                    'translation': final_translation,
                    'angle': final_angle,
                    'is_big_change': mask.shape[0] > 5,
                    'frame_diff': frame_diff,
                    'vid_names': str(self.imageset[index])
                }
            ]

        return {"images": rgbs, "cameras": cameras}

    def totrain(self, epoch):
        self.imageset = np.loadtxt(
            self.base_file + "/frames/%s/video_loc.txt" % "train", dtype=np.str
        )
        self.imageset = self.imageset[0 : int(0.8 * self.imageset.shape[0])]
        self.rng = np.random.RandomState(epoch)

    def toval(self, epoch):
        self.imageset = np.loadtxt(
            self.base_file + "/frames/%s/video_loc.txt" % "train", dtype=np.str
        )
        self.imageset = self.imageset[int(0.8 * self.imageset.shape[0]) :]
        self.rng = np.random.RandomState(epoch)


class RealEstate10KFixed(data.Dataset):
    """ Dataset for loading the fixed RealEstate10K test set -- images
    have already been randomly generated in realestate_test_indices, 
    simply select those sequentially here
    """

    def __init__(
        self, dataset, opts=None, num_views=2, seed=0, vectorize=False
    ):
        # Now go through the dataset

        self.imageset = np.loadtxt(
            opts.test_data_path + "/frames/%s/video_loc.txt" % "test",
            dtype=np.str,
        )
        self.base_file = opts.test_data_path

        self.indices=np.load('data/realestate_test_indices.npy')
        self.num_views = num_views

        self.dataset = "test"
        self.isTrain = False

        self.input_transform = Compose(
            [
                Resize((opts.W, opts.W)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.offset = np.array(
            [[2, 0, -1], [0, -2, 1], [0, 0, -1]],  # Flip ys to match habitat
            dtype=np.float32,
        )  # Make z negative to match habitat (which assumes a negative z)

        self.K = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.invK = np.linalg.inv(self.K)
        self.opts = opts
        print('val angle threshold: 15')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, image_number):
        image_indices = self.indices[image_number]

        frames = np.loadtxt(
                    self.base_file
                    + "/frames/%s/%s.txt" % (self.dataset, self.imageset[image_indices[0]]), skiprows=1
                )

        frame1 = frames[image_indices[1],  7:]
        frame2 = frames[image_indices[2],  7:]
        orig_viewpoint = frame1.reshape(3, 4)
        new_viewpoint = frame2.reshape(3, 4)
        angle, translation = get_deltas(orig_viewpoint, new_viewpoint)


        rgbs = []
        cameras = []
        for i in range(0, self.num_views):
            t_index = image_indices[i+1]
            image = Image.open(
                self.base_file
                + "/frames/%s/%s/" % (self.dataset, self.imageset[image_indices[0]])
            + str(int(frames[t_index, 0]))
                + ".jpg"
            )
            rgbs += [self.input_transform(image)]

            intrinsics = frames[t_index, 1:7]
            extrinsics = frames[t_index, 7:]

            origK = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            K = np.matmul(self.offset, origK)

            origP = extrinsics.reshape(3, 4)
            P = np.matmul(K, origP)  # Merge these together to match habitat
            P = np.vstack((P, np.zeros((1, 4), dtype=np.float32))).astype(
                np.float32
            )
            P[3, 3] = 1

            Pinv = np.linalg.inv(P)

            cameras += [
                {
                    "P": P,
                    "Pinv": Pinv,
                    "OrigP": origP,
                    "K": self.K,
                    "Kinv": self.invK,
                    'translation': translation,
                    'angle': angle,
                    'is_big_change': True,
                    'frame_diff': abs(image_indices[2]-image_indices[1]),
                    'vid_names': str(self.imageset[image_indices[0]])
                }
            ]

        return {"images": rgbs, "cameras": cameras}