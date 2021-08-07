# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as trn
import torchvision.models

from models.losses.synthesis import SynthesisLoss, PerceptualLoss
from models.networks.architectures import Unet
from models.networks.utilities import get_decoder, get_encoder
from models.projection.z_buffer_manipulator import PtsManipulator
from models.vqvae2.vqvae import VQVAETop
#from scipy.spatial.transform import Rotation as R

from models.lmconv.masking import get_generation_order_idx, get_masks
from models.lmconv.sample import sample
from models.lmconv.model import OurPixelCNN
from models.lmconv.layers import PONO
from models.lmconv.sample import sample
import mock 
import math
import cv2
import numpy as np
from PIL import Image

from models.layers.blocks import ResNet_Block

class ZbufferModelPts(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # ENCODER
        # Encode features to a given resolution
        self.encoder = get_encoder(opt)

        # POINT CLOUD TRANSFORMER
        # REGRESS 3D POINTS
        if hasattr(opt, 'Unet_num_filters'):
            self.pts_regressor = Unet(channels_in=3, channels_out=1, opt=opt, num_filters=opt.Unet_num_filters)
        else:
            self.pts_regressor = Unet(channels_in=3, channels_out=1, opt=opt)
        if "modifier" in self.opt.depth_predictor_type:
            self.modifier = Unet(channels_in=64, channels_out=64, opt=opt)

        # 3D Points transformer
        if self.opt.use_rgb_features:
            self.pts_transformer = PtsManipulator(opt.W, C=3, opt=opt)
        else:
            self.pts_transformer = PtsManipulator(opt.W, opt=opt)

        #self.directions = ['R','L']#,'U','D']
        #if 'diagonals' in self.opt and self.opt.diagonals:
        #    self.directions = ['R','L','U','D', 'UL', 'UR', 'DR', 'DL']

        # autoregressive outpaint
        if not "no_outpainting" in self.opt or not self.opt.no_outpainting:
            self.num_classes = 512
            norm_op = lambda num_channels: PONO()
            self.outpaint2 = OurPixelCNN(nr_resnet=2,
                nr_filters=80, 
                input_channels=self.num_classes,
                nr_logistic_mix=10,
                kernel_size=(3, 3),
                max_dilation=2,
                weight_norm=(False),
                feature_norm_op=norm_op,
                dropout_prob=0,
                conv_bias=(True),
                conv_mask_weight=False,
                rematerialize=False,
                binarize=False)
            self.args = mock.Mock 
            # used to revert to normal seed after sampling
            self.args.dataloader_seed = self.opt.seed
            
            self.obs = [3,32,32]
            self.upsample = nn.Upsample(scale_factor=8, mode="bilinear")
                
            self.vqvae = VQVAETop()
            self.ar_loss = nn.CrossEntropyLoss()

            self.args.num_classes = self.num_classes

            self.downsample = nn.AvgPool2d(kernel_size=8, stride=8)
            self.classifier = torchvision.models.__dict__['resnet18'](num_classes=365)

        self.projector = get_decoder(opt)

        # LOSS FUNCTION
        # Module to abstract away the loss function complexity
        self.loss_function = SynthesisLoss(opt=opt)

        self.min_tensor = self.register_buffer("min_z", torch.Tensor([0.1]))
        self.max_tensor = self.register_buffer(
            "max_z", torch.Tensor([self.opt.max_z])
        )
        self.discretized = self.register_buffer(
            "discretized_zs",
            torch.linspace(self.opt.min_z, self.opt.max_z, self.opt.voxel_size),
        )

        self.trn = trn.Compose([
                trn.Resize((224,224)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # instead of using given output rotation matrix, we use 
        # euler rotation of .6 as our full angle -- which we consider both left & right
        self.rotvecs = {'R': np.array([0,.6,0]), 'L': np.array([0,-.6,0]), 'U': np.array([-.3,0,0]), 'D': np.array([.3,0,0]), \
                    'UR': np.array([-.15,.3,0]), 'UL': np.array([-.15,-.3,0]), 'DR': np.array([.15,.3,0]), 'DL': np.array([.15,-.3,0])} 

        # used for consistency direction choice
        self.mapping = ['R','L','U','D', 'UL', 'UR', 'DR', 'DL']

    def process_batch(self, batch):
        # Input values
        input_img = batch["images"][0]
        
        # Camera parameters
        K = batch["cameras"][0]["K"]
        K_inv = batch["cameras"][0]["Kinv"]
        input_RT = batch["cameras"][0]["P"]
        input_RTinv = batch["cameras"][0]["Pinv"]
        
        if self.opt.model_setting == 'train' or self.opt.model_setting == 'gen_paired_img':
            output_img = batch["images"][-1]
            output_RT = batch["cameras"][-1]["P"]
            output_RTinv = batch["cameras"][-1]["Pinv"]
        elif self.opt.model_setting == 'get_gen_order':
            output_RT = batch["cameras"][-1]["P"]
            output_RTinv = batch["cameras"][-1]["Pinv"]
        elif self.opt.model_setting == 'gen_two_imgs':
            direction = batch["direction"]

        if torch.cuda.is_available():
            input_img = input_img.cuda()
            K = K.cuda()
            K_inv = K_inv.cuda()
            input_RT = input_RT.cuda()
            input_RTinv = input_RTinv.cuda()

            if self.opt.model_setting == 'train' or self.opt.model_setting == 'gen_paired_img':
                output_img = output_img.cuda()
                output_RT = output_RT.cuda()
                output_RTinv = output_RTinv.cuda()
            elif self.opt.model_setting == 'get_gen_order':
                output_RT = output_RT.cuda()
                output_RTinv = output_RTinv.cuda()
            elif self.opt.model_setting == 'gen_two_imgs':
                direction = direction.cuda()

        '''
        if self.opt.visualize_only and self.opt.dataset != 'realestate':
            # Input values
            input_img = batch["images"][0][0]
            output_img = batch["images"][-1][0]

            if "depths" in batch.keys():
                depth_img = batch["depths"][0][0]

            # Camera parameters
            K = batch["cameras"][0]["K"][0]
            K_inv = batch["cameras"][0]["Kinv"][0]

            input_RT = batch["cameras"][0]["P"][0]
            torch.set_printoptions(sci_mode=False)
            input_RTinv = batch["cameras"][0]["Pinv"][0]
            output_RT = batch["cameras"][-1]["P"][0]
            output_RTinv = batch["cameras"][-1]["Pinv"][0]
        '''
        
        if self.opt.model_setting == 'train' or self.opt.model_setting == 'gen_paired_img':
            return K, K_inv, input_RT, input_RTinv, output_RT, output_RTinv, input_img, output_img
        elif self.opt.model_setting == 'get_gen_order':
            return K, K_inv, input_RT, input_RTinv, output_RT, output_RTinv, input_img
        elif self.opt.model_setting == 'gen_two_imgs':
            return K, K_inv, input_RT, input_RTinv, input_img, direction
        else:
            return K, K_inv, input_RT, input_RTinv, input_img

    def eulerAnglesToRotationMatrix(self, theta) :
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R   

    def get_rt_from_rot(self, direction, input_RT, num=None, denom=None):
        # we convert to euler angles, modify, then go back to rotation matrix
        if self.opt.model_setting == 'gen_two_imgs' or self.opt.model_setting == 'gen_scene':
            if direction == 'S':
                new_output_RT = torch.zeros_like(input_RT).cuda()
                new_output_RT[:,:,:3] = input_RT[:,:,:3]
                new_output_RT[:,3,3] = 1
                new_output_RT[0,:3,3] = input_RT[0,:3,3] + .35 * torch.tensor([np.sin(2*np.pi*num/denom),np.cos(2*np.pi*num/denom),.4*np.sin(2*np.pi*(.25+num/denom))]).cuda()
                new_output_RTinv = torch.inverse(new_output_RT)
                return new_output_RTinv, new_output_RT
            elif direction == 'C':
                rotvec = np.array([0.2 * np.cos(2*np.pi*num/denom), 0.2 * np.sin(2*np.pi*num/denom),0])
                tmp_mtx = self.eulerAnglesToRotationMatrix(rotvec)
                mtx = torch.zeros([1,4,4]).cuda()
                mtx[0,3,3] = 1
                mtx[0,:3,:3] = torch.tensor(tmp_mtx).to(torch.float32).cuda()
                new_output_RT = mtx.bmm(input_RT)
                new_output_RTinv = torch.inverse(new_output_RT)
                return new_output_RTinv, new_output_RT
            else:
                rotvec = self.rotvecs[direction] * num / denom
        else:
            rotvec = self.rotvecs[direction] * self.opt.rotation / np.linalg.norm(self.rotvecs[direction])

        tmp_mtx = self.eulerAnglesToRotationMatrix(rotvec)
        mtx = torch.zeros([1,4,4]).cuda()
        mtx[0,3,3] = 1
        mtx[0,:3,:3] = torch.tensor(tmp_mtx).to(torch.float32).cuda()
        if self.opt.homography:
            new_output_RT = torch.zeros([1,4,4]).cuda()
            new_output_RT[:,:,3] = input_RT[:,:,3]
            new_output_RT[:,:3,:3] = mtx[:,:3,:3].bmm(input_RT[:,:3,:3])
        else:
            new_output_RT = mtx.bmm(input_RT)
        new_output_RTinv = torch.inverse(new_output_RT)
        return new_output_RTinv, new_output_RT

    def get_best_sample(self, gen_order, masks, downsampled_fs, background_mask, gen_fs, netD, input_img):
        discrim_scores, entropy_scores, imgs = [], [], []
        for i in range(self.opt.num_samples):                
            autoreg_output, autoreg_loss = sample(self.outpaint2, gen_order, *masks, downsampled_fs, \
                    self.obs, self.args, i, self.opt.temperature, self.downsample(background_mask.float()))
            autoreg_output = torch.argmax(autoreg_output, dim=1)
            ar_sample = self.vqvae.decode_code(autoreg_output.to(torch.int64).to('cuda'))
            combined = self.get_combined(gen_fs, ar_sample, background_mask)
            gen_img = self.projector(combined, background_mask)

            discrim_scores.append(netD.run_discriminator_one_step(gen_img, input_img)["D_Fake"].mean().cpu()) # use input so not using output

            png_stype = Image.fromarray(((gen_img[0].reshape([256,256,3]).cpu().numpy()*.5+.5)*255).astype(np.uint8))
            new_input_img = self.trn(png_stype)
            logit = self.classifier.forward(new_input_img.unsqueeze(0).cuda())
            h_x = F.softmax(logit.cpu(), 1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            entropy_scores.append(-np.sum(np.array(probs)*np.log(np.array(probs))))
            imgs.append(gen_img)

        entropy_ranks = []
        discrim_ranks = []
        sorted_disc = np.array(discrim_scores).argsort()
        sorted_entr = np.array(entropy_scores).argsort()
        for i in range(self.opt.num_samples):
            discrim_ranks.append(np.where(sorted_disc==i)[0][0])
            entropy_ranks.append(np.where(sorted_entr==i)[0][0])

        total = .5*(self.opt.num_samples-1-np.array(entropy_ranks))+.5*np.array(discrim_ranks)
        sorted_total = np.argsort(total)
        best = np.argmax(total)
        best_img = imgs[best]
        return best_img

    def forward(self, batch, netD=None):
        """ Forward pass of the view synthesis model.
        """

        if self.opt.model_setting == 'gen_scene' or self.opt.model_setting == 'gen_two_imgs':
            return self.forward_scene(batch, netD)
        elif self.opt.model_setting == 'get_gen_order':
            return self.forward_gen_order(batch)
        elif self.opt.model_setting == 'gen_paired_img' or self.opt.model_setting == 'gen_img':
            return self.forward_image(batch, netD)
        else:
            return self.forward_image(batch)

    def forward_image(self, batch, netD=None):
        """ single image forward pass
        """
        if self.opt.model_setting == 'train' or self.opt.model_setting == 'gen_paired_img':
            K, K_inv, input_RT, input_RTinv, output_RT, output_RTinv, input_img, output_img = self.process_batch(batch)
        else:
            K, K_inv, input_RT, input_RTinv, input_img = self.process_batch(batch)
            # get output_RT and output_RTinv
            output_RTinv, output_RT = self.get_rt_from_rot(self.opt.direction, input_RT)

        # Regressed points
        if not (self.opt.use_gt_depth):
            if not('use_inverse_depth' in self.opt) or not(self.opt.use_inverse_depth):
                regressed_pts = (
                    nn.Sigmoid()(self.pts_regressor(input_img))
                    * (self.opt.max_z - self.opt.min_z)
                    + self.opt.min_z
                )

            else:
                # Use the inverse for datasets with landscapes, where there
                # is a long tail on the depth distribution
                depth = self.pts_regressor(input_img)
                regressed_pts = 1. / (nn.Sigmoid()(depth) * 10 + 0.01)
        else:
            regressed_pts = depth_img

        if self.opt.use_rgb_features:
            fs = input_img
        else:
            fs = self.encoder(input_img)

        gen_fs, background_mask = self.pts_transformer.forward_justpts(
            fs,
            regressed_pts,
            K,
            K_inv,
            input_RT,
            input_RTinv,
            output_RT,
            output_RTinv,
        )

        if "modifier" in self.opt.depth_predictor_type:
            gen_fs = self.modifier(gen_fs)

        inner = None
        autoreg_loss = None
        if not "no_outpainting" in self.opt or not self.opt.no_outpainting:            
            masks_init, masks_undilated, masks_dilated, gen_order = self.get_masks_for_batch(output_RT, input_RTinv, background_mask)
            masks = [masks_init, masks_undilated, masks_dilated]

            if self.opt.model_setting == 'gen_img' or self.opt.model_setting == 'gen_paired_img':
                if 'vqvae' in self.opt and self.opt.vqvae:
                    downsampled_fs = self.vqvae.encode(gen_fs)[3]
                else:
                    downsampled_fs = self.downsample(gen_fs)
                
                gen_img = self.get_best_sample(gen_order, masks, downsampled_fs, background_mask, gen_fs, netD, input_img)
            else:
                # training: AR loss on downsampled GT input & output
                # decoder input is upsampled, downsampled GT & gen_fs
                downsampled_gt = self.downsample(output_img)
                if 'vqvae' in self.opt and self.opt.vqvae:
                    _, _, latent_loss, vqvae_encoded, _ = self.vqvae.encode(output_img)
                    if "pretrain" not in self.opt or not self.opt.pretrain:
                        downsampled_gt = (
                            F.one_hot(vqvae_encoded.detach(), self.num_classes).permute(0, 3, 1, 2).to(torch.float32)
                        )
                        ar_input = [downsampled_gt, masks_init, masks_undilated, masks_dilated]
                        autoreg_output = self.outpaint2(ar_input)
                        autoreg_loss = self.ar_loss(autoreg_output, vqvae_encoded)
                            
                else:
                    if "pretrain" not in self.opt or not self.opt.pretrain:
                        ar_input = [downsampled_gt, masks_init, masks_undilated, masks_dilated]
                        autoreg_output = self.outpaint2(ar_input)
                        autoreg_loss = discretized_mix_logistic_loss(downsampled_gt, autoreg_output, n_bits=8)
                # lambda doesn't matter -- not balancing these gradients with anything else.
                if 'vqvae' in self.opt and self.opt.vqvae:
                    # note vqvae is frozen 
                    input_gt = self.vqvae.decode_code(vqvae_encoded)
                else:
                    input_gt = self.upsample(downsampled_gt)
                    
                # input to decoder is combination of 
                # generated features (when in foreground)
                # and ground truth (when in background) as stand-in for autoreg. preds
                combined = self.get_combined(gen_fs, input_gt, background_mask)
                gen_img = self.projector(combined, background_mask)
                gen_fs = torch.cat((gen_fs, input_gt), 1)
        else:    
            gen_img = self.projector(gen_fs, inner)

        outputs = {
            "InputImg": input_img,
            "PredImg": gen_img,
            "PredDepthImg": regressed_pts / 5 - 1,
            'ForegroundImg': (~background_mask).repeat(input_img.shape[0],1,1,1).float()
        }

        if self.opt.model_setting == 'train':
            # And the loss
            loss = self.loss_function(gen_img, output_img)

            if (not "no_outpainting" in self.opt or not self.opt.no_outpainting) \
                    and ("pretrain" not in self.opt or not self.opt.pretrain):
                loss["autoreg_loss"] = autoreg_loss / (autoreg_output.shape[0] * np.prod(self.obs) * np.log(2.)) * 1000
                if 'lambda_autoreg' in self.opt and self.opt.lambda_autoreg is not None:
                    loss["Total Loss"] += autoreg_loss * self.opt.lambda_autoreg
                else:
                    loss["Total Loss"] += autoreg_loss

            if self.opt.train_depth:
                depth_loss = nn.L1Loss()(regressed_pts, depth_img)
                loss["Total Loss"] += depth_loss
                loss["depth_loss"] = depth_loss

            outputs["OutputImg"] = output_img
        elif self.opt.model_setting == 'gen_paired_img':
            outputs["OutputImg"] = output_img
            loss = self.loss_function(gen_img, gen_img) # loss is not used
        else:
            loss = self.loss_function(gen_img, gen_img) # loss is not used
        
        if self.opt.model_setting != 'train':
            outputs["FeaturesImg"] = gen_fs
            
        return (loss, outputs)

    def forward_scene(self, batch, netD=None):
        """ Forward pass to generate a scene (inference)
        """

        if self.opt.model_setting == 'gen_two_imgs':
            K, K_inv, input_RT, input_RTinv, input_img, direction = self.process_batch(batch)
            directions = [self.mapping[int(direction.cpu())]]
        else:
            K, K_inv, input_RT, input_RTinv, input_img = self.process_batch(batch)
            directions = self.opt.directions

        outputs = {
                "InputImg": input_img,
            }

        
        current_img = input_img
        
        last_background_mask = None
        prior_point_cloud = None
        fs_old = None
        last_output_RTinv = None
        last_numerator = None
        last_direction = None

        # first, we perform large completion
        # then, we smooth in everything between.
        # repeat for all four directions.
        # unless we are evaluating consistency, 
        # in which case we only look one direction
        for direction in directions:        
            num_split = self.opt.num_split
            if self.opt.model_setting == 'gen_two_imgs':
                num_split = 2
            elif direction in ['S','C']:
                num_split = self.opt.num_split * 2
            elif direction in ['U','D', 'UL', 'UR', 'DR', 'DL']:
                num_split = max(int(self.opt.num_split / 2),1)                

            print('Looking',direction+'.',num_split,'splits \n')

            # first do 0 case. if sampling, must do many times and pick best
            num_sample = 1
            if not "no_outpainting" in self.opt or not self.opt.no_outpainting:
                num_sample = self.opt.num_samples
                #if direction in ['U','D', 'UL', 'UR', 'DR', 'DL']:
                #    num_sample = max(int(num_sample / 2),1)
                best_loss = 0
                best_img = None
            
            if last_numerator is not None:
                current_input_RTinv, current_input_RT = self.get_rt_from_rot(last_direction, input_RT, last_numerator, num_split)
            else:
                current_input_RTinv, current_input_RT = input_RTinv, input_RT
            
            numerator = num_split
            current_output_RTinv, current_output_RT = self.get_rt_from_rot(direction, input_RT, numerator, num_split)

            regressed_pts = (
                    nn.Sigmoid()(self.pts_regressor(current_img))
                    * (self.opt.max_z - self.opt.min_z)
                    + self.opt.min_z
            )
            
            if self.opt.use_rgb_features:
                fs = current_img
            else:
                fs = self.encoder(current_img)
                
            gen_fs, background_mask, new_point_cloud, new_fs = self.pts_transformer.forward_justpts_cumulative(
                fs,
                regressed_pts,
                K,
                K_inv,
                current_input_RT,
                current_input_RTinv,
                current_output_RT,
                current_output_RTinv,
                prior_point_cloud,
                fs_old,
                last_background_mask,
                last_output_RTinv
            )

            if not "no_outpainting" in self.opt or not self.opt.no_outpainting:
                masks_init, masks_undilated, masks_dilated, gen_order = self.get_masks_for_batch(current_output_RT, current_input_RTinv, background_mask)
                masks = [masks_init, masks_undilated, masks_dilated]
                downsampled_fs = self.vqvae.encode(gen_fs)[3]
                
                best_img = self.get_best_sample(gen_order, masks, downsampled_fs, background_mask, gen_fs, netD, input_img)
            else:
                best_img = self.projector(gen_fs)

            gen_img = best_img
            current_img = gen_img
            prior_point_cloud = new_point_cloud
            fs_old = new_fs
            last_background_mask = background_mask
            last_output_RTinv = current_output_RTinv
            last_numerator = numerator
            last_direction = direction

            outputs['PredImg_'+direction+'_'+str(num_split)] = gen_img
            outputs["FeaturesImg_"+direction+'_'+str(num_split)] = gen_fs
            outputs["PredDepthImg_"+direction+'_'+str(num_split)] = regressed_pts
            outputs['ForegroundImg_'+direction+'_'+str(num_split)] = (~background_mask).repeat(input_img.shape[0],1,1,1).float()
            
            for i in reversed(range(0, num_split)):
                # we render an image at "0" to visualize input view in videos
                current_input_RTinv, current_input_RT = self.get_rt_from_rot(direction, input_RT, last_numerator, num_split)
                numerator = i
                current_output_RTinv, current_output_RT = self.get_rt_from_rot(direction, input_RT, numerator, num_split)
                if self.opt.use_rgb_features:
                    fs = current_img
                else:
                    fs = self.encoder(current_img)
                regressed_pts = (
                    nn.Sigmoid()(self.pts_regressor(current_img))
                    * (self.opt.max_z - self.opt.min_z)
                    + self.opt.min_z
                )

                gen_fs, background_mask, new_point_cloud, new_fs = self.pts_transformer.forward_justpts_cumulative(
                    fs,
                    regressed_pts,
                    K,
                    K_inv,
                    current_input_RT,
                    current_input_RTinv,
                    current_output_RT,
                    current_output_RTinv,
                    prior_point_cloud,
                    fs_old,
                    last_background_mask,
                    last_output_RTinv
                )

                if not "no_outpainting" in self.opt or not self.opt.no_outpainting:
                    masks_init, masks_undilated, masks_dilated, gen_order = self.get_masks_for_batch(current_output_RT, current_input_RTinv, background_mask)
                    masks = [masks_init, masks_undilated, masks_dilated]
                    downsampled_fs = self.vqvae.encode(gen_fs)[3]
                    gen_img = self.get_best_sample(gen_order, masks, downsampled_fs, background_mask, gen_fs, netD, input_img)
                else:
                    gen_img = self.projector(gen_fs)
                outputs['PredImg_'+direction+'_'+str(i)] = gen_img
                outputs["FeaturesImg_"+direction+'_'+str(i)] = gen_fs                    
                current_img = gen_img
                prior_point_cloud = new_point_cloud
                fs_old = new_fs
                last_background_mask = background_mask
                last_output_RTinv = current_output_RTinv
                last_numerator = numerator
            
        loss = self.loss_function(gen_img, gen_img) # loss not used
        return (loss, outputs)

    def forward_gen_order(self, batch):
        """ single image forward pass
        """
        K, K_inv, input_RT, input_RTinv, output_RT, output_RTinv, input_img = self.process_batch(batch)

        # Regressed points
        if not (self.opt.use_gt_depth):
            if not('use_inverse_depth' in self.opt) or not(self.opt.use_inverse_depth):
                regressed_pts = (
                    nn.Sigmoid()(self.pts_regressor(input_img))
                    * (self.opt.max_z - self.opt.min_z)
                    + self.opt.min_z
                )

            else:
                # Use the inverse for datasets with landscapes, where there
                # is a long tail on the depth distribution
                depth = self.pts_regressor(input_img)
                regressed_pts = 1. / (nn.Sigmoid()(depth) * 10 + 0.01)
        else:
            regressed_pts = depth_img

        if self.opt.use_rgb_features:
            fs = input_img
        else:
            fs = self.encoder(input_img)

        gen_fs, background_mask = self.pts_transformer.forward_justpts(
            fs,
            regressed_pts,
            K,
            K_inv,
            input_RT,
            input_RTinv,
            output_RT,
            output_RTinv,
        )
         
        masks_init, masks_undilated, masks_dilated, gen_order = self.get_masks_for_batch(output_RT, input_RTinv, background_mask)

        outputs = {
            'gen_order': torch.tensor(gen_order).cuda()
        }
        
        loss = self.loss_function(input_img, input_img) # loss not used
        return (loss, outputs)

    def get_masks_for_batch(self, output_RT, input_RTinv, background_mask):
        # 1 if in foreground, 0 if no points nearby
        foreground_mask = ~background_mask

        # downsample mask to be autoregressive size, convert bool to float
        background_mask = self.downsample(background_mask.float())
        foreground_mask = self.downsample(foreground_mask.float())

        b, h, w = background_mask.shape

        # multiply by index to get center of mass
        y=torch.arange(h).view(1,h,1)
        x=torch.arange(w).view(1,1,w)
        
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        mass_x = foreground_mask * x
        mass_y = foreground_mask * y

        mass_center_x = torch.mean(mass_x.view(b,-1),axis=1).cpu().detach()
        mass_center_y = torch.mean(mass_y.view(b,-1),axis=1).cpu().detach()
        mass_center = torch.stack((mass_center_x, mass_center_y), axis=1).numpy().astype(int)

        # get distance of each pixel to nearest background pixel
        # and distance of background pixels to nearest non-background pixel
        bin_fg_mask = foreground_mask.view(b, h, w, 1).cpu().detach().numpy().astype(np.uint8)
        bin_bg_mask = background_mask.view(b, h, w, 1).cpu().detach().numpy().astype(np.uint8)
        foreground_distances = np.zeros((b,h,w))
        background_distances = np.zeros((b,h,w))
        for image_num in range(b):
            foreground_distances[image_num] = cv2.distanceTransform(bin_fg_mask[image_num], distanceType=cv2.DIST_L2, maskSize=5)
            background_distances[image_num] = cv2.distanceTransform(bin_bg_mask[image_num], distanceType=cv2.DIST_L2, maskSize=5)
        distances = (foreground_distances - background_distances).astype(int)
        #example = np.array([[0,1,1,1],[1,1,1,1],[1,1,0,0],[0,0,0,0]]).astype(np.uint8)

        # autoregressive algorithm is as follows:
        # begin from maximum distance to background pixels
        # and proceed towards background pixels; fill in these
        # closest to foreground pixels, then furtherst
        # ties are broken using spiral pattern, 
        # which starts from center of mass.
        gen_order = []
        masks_init = []
        masks_undilated = []
        masks_dilated = []
        
        
        for image_num in range(b):
            gen_order.append(get_generation_order_idx('custom', self.obs[1], self.obs[2], distances[image_num], mass_center[image_num]))
            mask_init, mask_undilated, mask_dilated = get_masks(gen_order[-1], self.obs[1], self.obs[2], 3, 2, plot=False)#True, out_dir='log/mp3d/end_to_end_3x_upgraded_adjacent')
            masks_init.append(mask_init[0:1])
            masks_undilated.append(mask_undilated[0:1])
            masks_dilated.append(mask_dilated[0:1])

        masks_init = torch.stack(masks_init).repeat(1, 513, 1, 1).view(-1,9,self.obs[1]*self.obs[2]).cuda(non_blocking=True)
        masks_undilated = torch.stack(masks_undilated).repeat(1, 160, 1, 1).view(-1,9,self.obs[1]*self.obs[2]).cuda(non_blocking=True)
        masks_dilated = torch.stack(masks_dilated).repeat(1, 80, 1, 1).view(-1,9,self.obs[1]*self.obs[2]).cuda(non_blocking=True)
            
        return masks_init, masks_undilated, masks_dilated, gen_order

    def get_combined(self, gen_fs, ar_sample, background_mask):
        b, h, w = background_mask.shape
        foreground_mask = (~background_mask).float()
        background_mask = background_mask.float()
        combined = gen_fs * foreground_mask.view(b,-1,h,w) + ar_sample * background_mask.view(b,-1,h,w)
        return combined

    def forward_angle(self, batch, RTs, return_depth=False):
        # Input values
        input_img = batch["images"][0]

        # Camera parameters
        K = batch["cameras"][0]["K"]
        K_inv = batch["cameras"][0]["Kinv"]

        if torch.cuda.is_available():
            input_img = input_img.cuda()

            K = K.cuda()
            K_inv = K_inv.cuda()

            RTs = [RT.cuda() for RT in RTs]
            identity = (
                torch.eye(4).unsqueeze(0).repeat(input_img.size(0), 1, 1).cuda()
            )

        fs = self.encoder(input_img)
        regressed_pts = (
            nn.Sigmoid()(self.pts_regressor(input_img))
            * (self.opt.max_z - self.opt.min_z)
            + self.opt.min_z
        )

        # Now rotate
        gen_imgs = []
        for RT in RTs:
            torch.manual_seed(
                0
            )  # Reset seed each time so that noise vectors are the same
            gen_fs = self.pts_transformer.forward_justpts(
                fs, regressed_pts, K, K_inv, identity, identity, RT, None
            )

            # now create a new image
            gen_img = self.projector(gen_fs)

            gen_imgs += [gen_img]

        if return_depth:
            return gen_imgs, regressed_pts

        return gen_imgs
