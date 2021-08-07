# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch.nn as nn

from models.networks.architectures import (
    ResNetDecoder,
    ResNetEncoder,
    UNetDecoder64,
    UNetEncoder64,
)

EPS = 1e-2

def get_encoder(opt, downsample=True):
    if opt.refine_model_type == "unet":
        encoder = UNetEncoder64(opt, channels_in=3, channels_out=64)
    elif "resnet" in opt.refine_model_type:
        print("RESNET encoder")
        encoder = ResNetEncoder(
            opt, channels_in=3, channels_out=64, downsample=downsample
        )

    return encoder


def get_decoder(opt):
    if opt.refine_model_type == "unet":
        decoder = UNetDecoder64(opt, channels_in=64, channels_out=3)
    elif "resnet" in opt.refine_model_type:
        print("RESNET decoder")
        channels_in = 3
        if not "no_outpainting" in opt or not opt.no_outpainting:
            channels_in += 1
        decoder = ResNetDecoder(opt, channels_in=channels_in, channels_out=3)

    return decoder