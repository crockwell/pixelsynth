import logging

import numpy as np
import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("gen")


def configure_logger(filename="debug.log"):
    logger = logging.getLogger("gen")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("[%(asctime)s|%(name)s|%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    # Pytorch ordering
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis), inplace=True)


###########################
# Shared loss utilities
###########################

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def average_loss(log_probs_fn, x, ls, *xargs):
    """ ensemble multiple nn outputs (ls) by averaging likelihood """
    # Ensembles at the level of the joint distribution
    all_log_probs = []
    for l in ls:
        log_probs = log_probs_fn(x, l, *xargs)  # B x H x W x num_logistic_mix
        log_prob = log_sum_exp(log_probs)  # B x H x W
        log_prob = torch.sum(log_prob, dim=(1, 2))  # B, log prob of image under this
                                                    # ensemble component
        all_log_probs.append(log_prob)
    all_log_probs = torch.stack(all_log_probs, dim=1) - np.log(len(ls))  # B x len(ls)
    loss = -torch.sum(log_sum_exp(all_log_probs))
    return loss


###########################
# Multi-channel/color loss
###########################

def discretized_mix_logistic_log_probs(x, l, n_bits=8):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    assert n_bits > 0
    n_bins = 2. ** n_bits

    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]
   
    # here and below: unpacking the params of the mixture of logistics
    #import pdb 
    #pdb.set_trace()
    div=10
    # case of 4d
    if ls[-1] == 156:
        div=13
    elif ls[-1] == 372:
        div=31
    nr_mix = int(ls[-1] / div) 
    logit_probs = l[:, :, :, :nr_mix]
    
    if ls[-1] == 372:
        l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 5]) # 5: for mean, scale, need 3 coef
    else:
        l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
    #l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
   
    
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    if ls[-1] == 372:
        coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:5 * nr_mix])
        # need 16 coefficients for 6-dim correl mtx. 
        # To fit with rest of predictions,
        # we start as shape B,H,W,6,3*nr_mix
        # and reshape to B,H,W,16,nr_mix, and drop the remaining indices
        coeffs = coeffs.flatten()[:ls[0] * ls[1] * ls[2] * 16 * nr_mix].view([ls[0], ls[1], ls[2], 16, nr_mix])
        coeff_dims = [int(y) for y in coeffs.size()]
        #import pdb 
        #pdb.set_trace()
        #x = x.unsqueeze(-1) + Variable(torch.zeros(coeff_dims).cuda(), requires_grad=False)

    else:
        coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)

    m2 = (means[:, :, :, 1, :] + 
                coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + 
                coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    
    if ls[-1] == 156:
        # hope these coefficients are right!
        m4 = (means[:, :, :, 3, :] + 
                coeffs[:, :, :, 4, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 5, :] * x[:, :, :, 1, :] + 
                coeffs[:, :, :, 6, :] * x[:, :, :, 2, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
        means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3, m4), dim=3)
    elif ls[-1] == 372:
        div=31
        m4 = (means[:, :, :, 3, :] + 
                coeffs[:, :, :, 4, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 5, :] * x[:, :, :, 1, :] + 
                coeffs[:, :, :, 6, :] * x[:, :, :, 2, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
        m5 = (means[:, :, :, 4, :] + 
                coeffs[:, :, :, 7, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 8, :] * x[:, :, :, 1, :] + 
                coeffs[:, :, :, 9, :] * x[:, :, :, 2, :] +
                coeffs[:, :, :, 10, :] * x[:, :, :, 3, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
        m6 = (means[:, :, :, 5, :] + 
                coeffs[:, :, :, 11, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 12, :] * x[:, :, :, 1, :] + 
                coeffs[:, :, :, 13, :] * x[:, :, :, 2, :] +
                coeffs[:, :, :, 14, :] * x[:, :, :, 3, :] +
                coeffs[:, :, :, 15, :] * x[:, :, :, 4, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
        means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3, m4, m5, m6), dim=3)
    else:
        means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / (n_bins - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / (n_bins - 1))
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value
    
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return log_probs


def discretized_mix_logistic_loss(x, l, n_bits=8):
    """ reduced (summed) log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
    
    Args:
        x: B x C x H x W ground truth image
        l: B x (10 * num_logistic_mix) x H x W output of NN

    Returns:
        loss: 0-dimensional NLL loss tensor
    """
    log_probs = discretized_mix_logistic_log_probs(x, l, n_bits)  # B x H x W x num_logistic_mix
    return -torch.sum(log_sum_exp(log_probs))
 

def discretized_mix_logistic_loss_averaged(x, ls, n_bits=8):
    """ reduced (summed) log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
    Averages likelihood across multiple sets of mixture parameters
    
    Args:
        x: B x C x H x W ground truth image
        ls: list of B x (10 * num_logistic_mix) x H x W outputs of NN

    Returns:
        loss: 0-dimensional NLL loss tensor
    """
    return average_loss(discretized_mix_logistic_log_probs, x, ls, n_bits)


###################
# 1D (1 color) loss
###################

def discretized_mix_logistic_log_probs_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # 2 for mean, scale
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return log_probs


def discretized_mix_logistic_loss_1d(x, l):
    """ reduced (summed) log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
    
    Args:
        x: B x C x H x W ground truth image
        l: B x (3 * num_logistic_mix) x H x W output of NN
    """
    log_probs = discretized_mix_logistic_log_probs_1d(x, l)
    return -torch.sum(log_sum_exp(log_probs))


def discretized_mix_logistic_loss_1d_averaged(x, ls):
    """ reduced (summed) log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
    Averages likelihood across multiple sets of mixture parameters
    
    Args:
        x: B x C x H x W ground truth image
        ls: list of B x (3 * num_logistic_mix) x H x W outputs of NN
    """
    return average_loss(discretized_mix_logistic_log_probs_1d, x, ls)


######################################################
# Binarization utilities and cross entropy losses
######################################################

def _binarized_label(x):
    assert x.size(1) == 1
    x = x * .5 + .5  # Scale from [-1, 1] to [0, 1] range
    x = binarize_torch(x)  # binarize image. Should be able to just cast,
                            # since x is either 0. or 1., but this could avoid float
                            # innacuracies from rescaling.
    x = x.squeeze(1).long()
    return x


def _binarized_log_probs(x, l):
    """Cross-entropy loss

    Args:
        x: B x H x W floating point ground truth image, [-1, 1] scale
        l: B x 2 x H x W output of neural network

    Returns:
        log_probs: B x H x W x 1 tensor of likelihod of each pixel in x
    """
    assert l.size(1) == 2
    x = _binarized_label(x)
    l = F.log_softmax(l, dim=1)
    log_probs = -F.nll_loss(l, x, reduction="none").unsqueeze(-1)
    return log_probs


def binarized_loss(x, l):
    """Cross-entropy loss

    Args:
        x: B x 1 x H x W floating point ground truth image, [-1, 1] scale
        l: B x 2 x H x W output of neural network

    Returns:
        loss: 0-dimensional NLL loss tensor
    """
    assert l.size(1) == 2
    x = _binarized_label(x)
    # cross_entropy averages across the batch, so we multiply by batch size
    # to keep a similar loss scale as with grayscale MNIST
    return F.cross_entropy(l, x, reduction="sum")


def binarized_loss_averaged(x, ls):
    """
    Args:
        x: B x C x H x W ground truth image
        ls: list of B x 2 x H x W outputs of NN

    Returns:
        loss: 0-dimensional NLL loss tensor
    """
    return average_loss(_binarized_log_probs, x, ls)


def binarize_np(images: np.ndarray):
    rand = np.random.uniform(size=images.shape)
    return (rand < images).astype(np.float32)


def binarize_torch(images):
    rand = torch.rand(images.shape, device=images.device)
    return (rand < images).float()


###########
# Sampling
###########

def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda : one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)


def sample_from_discretized_mix_logistic_1d(l, coord1, coord2, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1] #[3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # for mean, scale

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda : temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)
   
    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4) 
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    u = torch.FloatTensor(means.size())
    if l.is_cuda : u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out.data[:, :, coord1, coord2]


def sample_from_discretized_mix_logistic(l, coord1, coord2, nr_mix, mixture_temperature=1.0, logistic_temperature=1.0, temp_eps=.05, get_likelihood=False, seed=None):
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    div=10
    # case of 4d
    if ls[-1] == 156:
        xs = ls[:-1] + [4]
        div=13
        nr_mix = int(ls[-1] / div) 
    elif ls[-1] == 372:
        xs = ls[:-1] + [6]
        div=31
        nr_mix = int(ls[-1] / div) 
    else:
        xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    if ls[-1] == 372:
        l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 5]) # 5: for mean, scale, need 3 coef
    else:
        l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda : temp = temp.cuda()
    #temp.uniform_(1e-5, 1. - 1e-5)
    temp.uniform_(temp_eps, 1-temp_eps)
    if seed is not None:
        for i in range(seed):
            temp.uniform_(temp_eps, 1-temp_eps)
    
    temp = logit_probs.data - torch.log(- torch.log(temp)) * mixture_temperature
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4) 
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    if ls[-1] == 372:
        coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:5 * nr_mix])
        # need 16 coefficients for 6-dim correl mtx. 
        # To fit with rest of predictions,
        # we start as shape B,H,W,6,3*nr_mix
        # and reshape to B,H,W,16,nr_mix, and drop the remaining indices
        coeffs = coeffs.flatten()[:ls[0] * ls[1] * ls[2] * 16 * nr_mix].view([ls[0], ls[1], ls[2], 16, nr_mix])
        coeffs = torch.sum(coeffs * sel, dim=4)
    else:
        coeffs = torch.sum(torch.tanh(
            l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda : u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    if seed is not None:
        for i in range(seed):
            u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * logistic_temperature * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(
       x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = torch.clamp(torch.clamp(
       x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)
    
    if ls[-1] == 156:
        # hope these coefficients are right! 
        x3 = torch.clamp(torch.clamp(
            x[:, :, :, 3] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1 + coeffs[:, :, :, 3] * x2, min=-1.), max=1.)
        out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1]), x3.view(xs[:-1] + [1])], dim=3)
    elif ls[-1] == 372:
        x3 = torch.clamp(torch.clamp((x[:, :, :, 3] + 
                coeffs[:, :, :, 4] * x0 +
                coeffs[:, :, :, 5] * x1 + 
                coeffs[:, :, :, 6] * x2), min=-1.), max=1.)
        x4 = torch.clamp(torch.clamp((x[:, :, :, 4] + 
                coeffs[:, :, :, 7] * x0 +
                coeffs[:, :, :, 8] * x1 + 
                coeffs[:, :, :, 9] * x2 +
                coeffs[:, :, :, 10] * x3), min=-1.), max=1.)
        x5 = torch.clamp(torch.clamp((x[:, :, :, 5] + 
                coeffs[:, :, :, 11] * x0 +
                coeffs[:, :, :, 12] * x1 + 
                coeffs[:, :, :, 13] * x2 +
                coeffs[:, :, :, 14] * x3 +
                coeffs[:, :, :, 15] * x4), min=-1.), max=1.)
        out = torch.cat((x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1]), \
                    x3.view(xs[:-1] + [1]), x4.view(xs[:-1] + [1]), x5.view(xs[:-1] + [1])), dim=3)
    else:
        out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)

    if get_likelihood:
        # calc likelihood avg across rgb, based on each distribution.
        # only consider selected logistic distribution.
        from torch.distributions.transformed_distribution import TransformedDistribution
        from torch.distributions.transforms import SigmoidTransform, AffineTransform
        from torch.distributions import Uniform
        #predictions = out.data[0, coord1, coord2, :]
        predictions = x.data[0, coord1, coord2, :]
        this_mean = means[0, coord1, coord2, :]
        this_scale = torch.exp(log_scales[0, coord1, coord2, :])
        base_distribution = Uniform(0, 1)
        transforms_r = [SigmoidTransform().inv, AffineTransform(loc=this_mean[0], scale=this_scale[0])]
        logistic_r = TransformedDistribution(base_distribution, transforms_r)
        log_prob_r = logistic_r.log_prob(predictions[0])

        transforms_g = [SigmoidTransform().inv, AffineTransform(loc=this_mean[1], scale=this_scale[1])]
        logistic_g = TransformedDistribution(base_distribution, transforms_g)
        log_prob_g = logistic_g.log_prob(predictions[1])

        transforms_b = [SigmoidTransform().inv, AffineTransform(loc=this_mean[2], scale=this_scale[2])]
        logistic_b = TransformedDistribution(base_distribution, transforms_b)
        log_prob_b = logistic_b.log_prob(predictions[2])

        log_prob=(log_prob_r + log_prob_g + log_prob_b)/3

        return out.data[:, coord1, coord2, :], log_prob

    if coord1 is None:
        return out.data
    else:
        return out.data[:, coord1, coord2, :]


def sample_from_binary_logits(l, coord1, coord2):
    """
    Args:
        l: B x 2 x H x W output of NN (logits)
        coord1
        coord2

    Returns:
        pixels: B x 1 pixel samples at location (coord1, coord2) in range [-1, 1]
    """
    assert l.size(1) == 2
    l = l[:, :, coord1, coord2]
    pixels = torch.distributions.categorical.Categorical(logits=l).sample()
    pixels = pixels * 2. - 1.
    return pixels.unsqueeze(1)


#########################################################################################
# utilities for shifting the image around, efficient alternative to masking convolutions
#########################################################################################

def down_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when downshifting, the last row is removed 
    x = x[:, :, :xs[2] - 1, :]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when righshifting, the last column is removed 
    x = x[:, :, :, :xs[3] - 1]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)


#######################
# Restoring checkpoint
#######################

def load_part_of_model(path, model, optimizer=None):
    checkpoint = torch.load(path)
    params = checkpoint["model_state_dict"]
    # Restore model
    logger.info("Restoring model from %s", path)
    added = 0
    for name, param in params.items():
        if name in model.state_dict().keys():
            try:
                model.state_dict()[name].copy_(param)
                added += 1
            except Exception as e:
                logger.warning("Error loading model.state_dict()[%s]: %s", name, e)
        else:
            logger.warning("Key present in checkpoint that is not present in model.state_dict(): %s", name)
    logger.info('Loadded %s fraction of params:' % (added / float(len(model.state_dict().keys()))))

    # Restore optimizer
    if optimizer:
        logger.info("Restoring optimizer from %s", path)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info('Loaded optimizer params directly')
        except Exception as e:
            logger.warning("Failed to load entire optimizer state dict at once, trying each key of state only")

            added = 0
            for name, param in checkpoint["optimizer_state_dict"]["state"].items():
                if name in optimizer.state_dict()["state"].keys():
                    try:
                        optimizer.state_dict()["state"][name].copy_(param)
                        added += 1
                    except Exception as e:
                        logger.error("Error loading optimizer.state_dict()['state'][%s]: %s", name, e)
                        pass
            logger.info('Loaded %s fraction of optimizer params:' % (added / float(len(optimizer.state_dict()["state"].keys()))))

            # TODO: load param_groups key?

    return checkpoint["epoch"], checkpoint.get("global_step", -1)


class EMA():
    # Computes exponential moving average of model parameters, adapted from https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/3
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, model):
        for name, param in model.state_dict().items():
            self.shadow[name] = param.clone()

    def update(self, model):
        for name, param in model.state_dict().items():
            assert name in self.shadow
            new_average = self.mu * param + (1.0 - self.mu) * self.shadow[name]
            self.shadow[name] = new_average.clone()
            return new_average

    def state_dict(self):
        return self.shadow
