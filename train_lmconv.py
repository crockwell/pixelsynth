from IPython import embed
import argparse
import itertools
from operator import itemgetter
import os
import re
import time

from PIL import Image
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
import tqdm

#from baseline import PixelCNN
from models.lmconv.layers import PONO
from models.lmconv.masking import *
from models.lmconv.model import OurPixelCNN
from models.lmconv.utils import *
from models.vqvae2.vqvae import VQVAETop
import pickle as pkl

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4"
N_CLASS=512
TEMPERATURE=1

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist|celeba')
parser.add_argument('--binarize', action='store_true')
parser.add_argument('-p', '--print_every', type=int, default=20,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=20,
                    help='Every how many epochs to write checkpoint?')
parser.add_argument('-ts', '--sample_interval', type=int, default=4,
                    help='Every how many epochs to write samples?')
parser.add_argument('-tt', '--test_interval', type=int, default=1,
                    help='Every how many epochs to test model?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
parser.add_argument('--load_last_params', action="store_true",
                    help='Restore training from the last model checkpoint in the run dir?')
parser.add_argument('--do_not_load_optimizer', action="store_true")
parser.add_argument('-rd', '--run_dir', type=str, default=None,
                    help="Optionally specify run directory. One will be generated otherwise."
                         "Use to save log files in a particular place")
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('-ID', '--exp_id', type=int, default=0)
parser.add_argument('--ours', action='store_true')
parser.add_argument('--dset_dir', type=str, default='models/vqvae2/embeddings',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('--vqvae_path', type=str, default='',
                    help='Location for vqvae checkpoint - used for sampling')
parser.add_argument('--gen_order_dir', type=str, default='data',
                    help='Location for parameter checkpoints and samples')
# only for CelebAHQ
parser.add_argument('--max_celeba_train_batches', type=int, default=-1)
parser.add_argument('--max_celeba_test_batches', type=int, default=-1)
parser.add_argument('--celeba_size', type=int, default=256)
parser.add_argument('--n_bits', type=int, default=8)
# pixelcnn++ and our model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-wd', '--weight_decay', type=float,
                    default=0, help='Weight decay during optimization')
parser.add_argument('-c', '--clip', type=float, default=-1, help='Gradient norms clipped to this value')
parser.add_argument('-b', '--batch_size', type=int, default=64,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('--ema', type=float, default=1)
# our model
parser.add_argument('-k', '--kernel_size', type=int, default=5,
                    help='Size of conv kernels')
parser.add_argument('-md', '--max_dilation', type=int, default=2,
                    help='Dilation in downsize stream')
parser.add_argument('-dp', '--dropout_prob', type=float, default=0.5,
                    help='Dropout prob used with nn.Dropout2d in gated resnet layers. '
                         'Argument only used if --ours is provided. Set to 0 to disable '
                         'dropout entirely.')
parser.add_argument('-nm', '--normalization', type=str, default='weight_norm',
                    choices=["none", "weight_norm", "order_rescale", "pono"])
parser.add_argument('-af', '--accum_freq', type=int, default=1,
                    help='Batches per optimization step. Used for gradient accumulation')
parser.add_argument('--two_stream', action="store_true", help="Enable two stream model")
parser.add_argument('--order', type=str, nargs="+",
                    choices=["raster_scan", "s_curve", "hilbert", "gilbert2d", "s_curve_center_quarter_last", 'custom'],
                    help="Autoregressive generation order")
parser.add_argument('--randomize_order', action="store_true", help="Randomize between 8 variants of the "
                    "pixel generation order.")
parser.add_argument('--mode', type=str, choices=["train", "sample", "test", "count_params"],
                    default="train")
# configure training
parser.add_argument('--train_masks', nargs="*", type=int, help="Specify indices of masks in all_masks to use during training")
# configure sampling
parser.add_argument('--sample_region', type=str, choices=["full", "center", "random_near_center", "top", "custom"], default="full")
parser.add_argument('--sample_size_h', type=int, default=16, help="Only used for --sample_region center, top or random. =H of inpainting region.")
parser.add_argument('--sample_size_w', type=int, default=16, help="Only used for --sample_region center, top or random. =W of inpainting region.")
parser.add_argument('--sample_offset1', type=int, default=None, help="Manually specify box offset for --sample_region custom")
parser.add_argument('--sample_offset2', type=int, default=None, help="Manually specify box offset for --sample_region custom")
parser.add_argument('--sample_batch_size', type=int, default=25, help="Number of images to sample")
parser.add_argument('--sample_mixture_temperature', type=float, default=1.0)
parser.add_argument('--sample_logistic_temperature', type=float, default=1.0)
parser.add_argument('--sample_quantize', action="store_true", help="Quantize images during sampling to avoid train-sample distribution shift")
parser.add_argument('--save_nrow', type=int, default=4)
parser.add_argument('--save_padding', type=int, default=2)
# configure testing
parser.add_argument('--test_masks', nargs="*", type=int, help="Specify indices of masks in all_masks to use during testing")
parser.add_argument('--test_region', type=str, choices=["full", "custom"], default="full")
parser.add_argument('--test_minh', type=int, default=0, help="Specify conditional likelihood testing region. Only used with --test_region custom")
parser.add_argument('--test_maxh', type=int, default=32, help="Specify conditional likelihood testing region. Only used with --test_region custom")
parser.add_argument('--test_minw', type=int, default=0, help="Specify conditional likelihood testing region. Only used with --test_region custom")
parser.add_argument('--test_maxw', type=int, default=32, help="Specify conditional likelihood testing region. Only used with --test_region custom")
parser.add_argument('--order_variants', nargs="*", type=int)
# our model
parser.add_argument('--no_bias', action="store_true", help="Disable learnable bias for all convolutions")
parser.add_argument('--learn_weight_for_masks', action="store_true", help="Condition each masked conv on the mask itself, with a learned weight")
parser.add_argument('--minimize_bpd', action="store_true", help="Minimize bpd, scaling loss down by number of dimension")
parser.add_argument('--resize_sizes', type=int, nargs="*")
parser.add_argument('--resize_probs', type=float, nargs="*")
parser.add_argument('--base_order_reflect_rows', action="store_true")
parser.add_argument('--base_order_reflect_cols', action="store_true")
parser.add_argument('--base_order_transpose', action="store_true")
# memory and precision
parser.add_argument('--rematerialize', action="store_true", help="Recompute some activations during backwards to save memory")
parser.add_argument('--amp_opt_level', type=str, default=None)
# plotting
parser.add_argument('--plot_masks', action="store_true")

args = parser.parse_args()


# Set seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Create run directory
if args.run_dir:
    run_dir = args.run_dir
    try: 
        os.makedirs(run_dir)
    except:
        pass
else:
    dataset_name = args.dataset if not args.binarize else f"binary_{args.dataset}"
    _name = "{:05d}_{}_lr{:.5f}_bs{}_gc{}_k{}_md{}".format(
        args.exp_id, dataset_name, args.lr, args.batch_size, args.clip, args.kernel_size, args.max_dilation)
    if args.normalization != "none":
        _name = f"{_name}_{args.normalization}"
    if args.exp_name:
        _name = f"{_name}+{args.exp_name}"
    run_dir = os.path.join("runs", _name)
    if args.mode == "train":
        os.makedirs(run_dir, exist_ok=args.load_last_params)

assert os.path.exists(run_dir), "Did not find run directory, check --run_dir argument"

# Log arguments
timestamp = time.strftime("%Y%m%d-%H%M%S")
if args.mode == "test" and args.test_region == "custom":
    logfile = f"{args.mode}_{args.test_minh}:{args.test_maxh}_{args.test_minw}:{args.test_maxw}_{timestamp}.log"
else:
    logfile = f"{args.mode}_{timestamp}.log"
logger = configure_logger(os.path.join(run_dir, logfile))
logger.info("Run directory: %s", run_dir)
logger.info("Arguments: %s", args)
for k, v in vars(args).items():
    logger.info(f"  {k}: {v}")


# Create data loaders
sample_batch_size = args.sample_batch_size
if 'mp3d' in args.dataset or 'realestate' in args.dataset:
    res_ = 32
dataset_obs = {
    'mnist': (1, 28, 28),
    'cifar': (3, 32, 32),
    'celebahq': (3, args.celeba_size, args.celeba_size),
    'mp3d': (1, 32, 32),
    'realestate': (1,32,32),
}[args.dataset]
input_channels = dataset_obs[0]
data_loader_kwargs = {'num_workers':4, 'pin_memory':True, 'drop_last':True, 'batch_size':args.batch_size}
data_sampler_kwargs = {'num_workers':4, 'pin_memory':True, 'drop_last':True, 'batch_size':args.sample_batch_size}
if args.resize_sizes:
    if not args.resize_probs:
        args.resize_probs = [1. / len(args.resize_sizes)] * len(args.resize_sizes)
    assert len(args.resize_probs) == len(args.resize_sizes)
    assert sum(args.resize_probs) == 1
    resized_obses = [(input_channels, s, s) for s in args.resize_sizes]
else:
    args.resize_sizes = [dataset_obs[1]]
    args.resize_probs = [1.]
    resized_obses = [dataset_obs]

def obs2str(obs):
    return 'x'.join(map(str, obs))

def random_resized_obs():
    idx = np.arange(len(resized_obses))
    obs_i = np.random.choice(idx, p=args.resize_probs)
    return resized_obses[int(obs_i)]

def get_resize_collate_fn(obs, default_collate=torch.utils.data.dataloader.default_collate):
    if obs == dataset_obs:
        return default_collate

    def resize_collate_fn(batch):
        X, y = default_collate(batch)
        X = torch.nn.functional.interpolate(X, size=obs[1:], mode="bilinear")
        return [X, y]
    return resize_collate_fn

def random_resize_collate(batch, default_collate=torch.utils.data.dataloader.default_collate):
    X, y = default_collate(batch)
    obs = random_resized_obs()
    if obs != dataset_obs:
        X = torch.nn.functional.interpolate(X, size=obs[1:], mode="bilinear")
    return [X, y]

# Create data loaders
if 'mnist' in args.dataset :
    assert args.n_bits == 8
    if args.binarize:
        rescaling = lambda x : (binarize_torch(x) - .5) * 2.  # binarze and rescale [0, 1] images into [-1, 1] range
    else:
        rescaling = lambda x : (x - .5) * 2.  # rescale [0, 1] images into [-1, 1] range
    rescaling_inv = lambda x : .5 * x + .5
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True,
        train=True, transform=ds_transforms), shuffle=True, collate_fn=random_resize_collate, **data_loader_kwargs)
    test_loader_by_obs = {
        obs: torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False,
            transform=ds_transforms), collate_fn=get_resize_collate_fn(obs), **data_loader_kwargs)
        for obs in resized_obses
    }

    # Default upper bounds for progress bars
    train_total = None
    test_total = None
elif 'cifar' in args.dataset :
    assert args.n_bits == 8
    rescaling = lambda x : (x - .5) * 2.  # rescale [0, 1] images into [-1, 1] range
    rescaling_inv = lambda x : .5 * x + .5
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), shuffle=True, collate_fn=random_resize_collate, **data_loader_kwargs)
    test_loader_by_obs = {
        obs: torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False,
            transform=ds_transforms), collate_fn=get_resize_collate_fn(obs), **data_loader_kwargs)
        for obs in resized_obses
    }

    # Default upper bounds for progress bars
    train_total = None
    test_total = None
elif 'mp3d' in args.dataset or 'realestate' in args.dataset:
    assert args.n_bits == 8
    rescaling = lambda x : (x - .5) * 2.  # rescale [0, 1] images into [-1, 1] range
    rescaling_inv = lambda x : .5 * x + .5
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

    class DispDataset(torch.utils.data.Dataset):
        def __init__(self, vals, orders):
            super(DispDataset, self).__init__()
            self.vals = {}
            i = 0
            ctvals = {}
            for val in orders:
                self.vals[val] = (torch.from_numpy(vals[val]).unsqueeze(0), orders[val])
            
            self.length = len(self.vals)

        def __getitem__(self, i):
            # we pair each with a y value, blank, to fit
            # with other dataloaders
            while i % len(self.vals) not in self.vals: 
                i = i * 3 - 1
            return (self.vals[i % len(self.vals)], 0)

        def __len__(self):
            return self.length

    if 'mp3d' in args.dataset:
        data_train_ = np.load(os.path.join(args.dset_dir,'mp3d_train.npy'))
        with open(os.path.join(args.gen_order_dir,'mp3d_train_gen_order.pkl'), 'rb') as f:
            gen_order_train = pkl.load(f)
        data_val_ = np.load(os.path.join(args.dset_dir,'mp3d_val.npy'))
        with open(os.path.join(args.gen_order_dir,'mp3d_val_gen_order.pkl'), 'rb') as f:
            gen_order_val = pkl.load(f)
    else:
        data_train_ = np.load(os.path.join(args.dset_dir,'realestate_train.npy'))
        with open(os.path.join(args.gen_order_dir,'realestate_train_gen_order.pkl'), 'rb') as f:
            gen_order_train = pkl.load(f)
        data_val_ = np.load(os.path.join(args.dset_dir,'realestate_val.npy'))
        with open(os.path.join(args.gen_order_dir,'realestate_val_gen_order.pkl'), 'rb') as f:
            gen_order_val = pkl.load(f)
    
    data_train = DispDataset(data_train_, gen_order_train)
    data_val = DispDataset(data_val_, gen_order_val)

    train_loader = torch.utils.data.DataLoader(data_train, shuffle=True, \
        collate_fn=random_resize_collate, **data_loader_kwargs)
    test_loader_by_obs = {
        obs: torch.utils.data.DataLoader(data_val, collate_fn=get_resize_collate_fn(obs), \
            **data_loader_kwargs)
        for obs in resized_obses
    } 
    sample_loader_by_obs = {
        obs: torch.utils.data.DataLoader(data_val, collate_fn=get_resize_collate_fn(obs), \
            **data_sampler_kwargs)
        for obs in resized_obses
    }
    
    # Default upper bounds for progress bars
    train_total = len(data_train_) // args.batch_size
    test_total = len(data_val_) // args.batch_size
elif 'celebahq' in args.dataset :
    if args.n_bits == 8:
        rescaling = lambda x : (2. / 255) * x - 1.  # rescale uint8 images into [-1, 1] range
        rescaling_inv = lambda x : .5 * x + .5  # rescale [-1, 1] range to [0, 1] range
    else:
        assert 0 < args.n_bits < 8
        n_bins = 2. ** args.n_bits
        depth_divisor = (2. ** (8 - args.n_bits))
        def rescaling(x):
            # reduce bit depth, from [0, 255] to [0, n_bins-1] range
            x = torch.floor(x / depth_divisor)
            # rescale images from [0, n_bins-1] into [-1, 1] range
            x = (2. / (n_bins - 1)) * x - 1.
            return x
        rescaling_inv = lambda x : .5 * x + .5  # rescale [-1, 1] range to [0, 1] range

    # NOTE: Random resizing of images during training is not supported for CelebA-HQ. Will use 256x256 resolution.
    def get_celeba_dataloaders():
        from celeba_data import get_celeba_dataloader
        kwargs = dict(data_loader_kwargs)
        kwargs["num_workers"] = 0
        train_loader = get_celeba_dataloader(args.data_dir, "train",
                                            collate_fn=itemgetter(0), # lambda batch: random_resize_collate(batch, itemgetter(0)),
                                            batch_transform=rescaling,
                                            max_batches=args.max_celeba_train_batches,
                                            size=args.celeba_size,
                                            **kwargs)
        test_loader_by_obs = {
            obs: get_celeba_dataloader(args.data_dir, "validation",
                                    collate_fn=get_resize_collate_fn(obs, itemgetter(0)),
                                    batch_transform=rescaling,
                                    max_batches=args.max_celeba_test_batches,
                                    size=args.celeba_size,
                                    **kwargs)
            for obs in resized_obses
        }
        return train_loader, test_loader_by_obs

    train_loader, test_loader_by_obs = get_celeba_dataloaders()

    # Manually specify upper bounds for progress bars
    train_total = 27000 // args.batch_size if args.max_celeba_train_batches <= 0 else args.max_celeba_train_batches
    test_total = 3000 // args.batch_size if args.max_celeba_test_batches <= 0 else args.max_celeba_test_batches
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))


def quantize(x):
    # Quantize [-1, 1] images to uint8 range, then put back in [-1, 1]
    # Can be used during sampling with --sample_quantize argument
    assert args.n_bits == 8
    continuous_x = rescaling_inv(x) * 255  # Scale to [0, 255] range
    discrete_x = continuous_x.long().float()  # Round down
    quantized_x = discrete_x / 255.
    return rescaling(quantized_x)


# Select loss functions
if 'mnist' in args.dataset or 'mp3d' in args.dataset or 'realestate' in args.dataset:
    # Losses for 1-channel images
    assert args.n_bits == 8
    if args.binarize:
        loss_op = binarized_loss
        loss_op_averaged = binarized_loss_averaged
        sample_op = sample_from_binary_logits
    else:
        loss_op = nn.CrossEntropyLoss()
        loss_op_averaged = nn.CrossEntropyLoss()
        sample_op = lambda x, i, j: torch.multinomial(x, 1).squeeze(-1)
else:
    # Losses for 3-channel images
    loss_op = lambda x, l : discretized_mix_logistic_loss(x, l, n_bits=args.n_bits)
    loss_op_averaged = lambda x, ls : discretized_mix_logistic_loss_averaged(x, ls, n_bits=args.n_bits)
    sample_op = lambda x, i, j: sample_from_discretized_mix_logistic(x, i, j, args.nr_logistic_mix,
                                                                              args.sample_mixture_temperature,
                                                                              args.sample_logistic_temperature)


# Construct model
if args.ours:
    logger.info("Constructing our model")

    if args.normalization == "order_rescale":
        norm_op = lambda num_channels: OrderRescale()
    elif args.normalization == "pono":
        norm_op = lambda num_channels: PONO()
    else:
        norm_op = None

    assert not args.two_stream, "--two_stream cannot be used with --ours"
    model_sample = OurPixelCNN(
                nr_resnet=args.nr_resnet,
                nr_filters=args.nr_filters, 
                input_channels=N_CLASS,
                nr_logistic_mix=args.nr_logistic_mix,
                kernel_size=(args.kernel_size, args.kernel_size),
                max_dilation=args.max_dilation,
                weight_norm=(args.normalization == "weight_norm"),
                feature_norm_op=norm_op,
                dropout_prob=args.dropout_prob,
                conv_bias=(not args.no_bias),
                conv_mask_weight=args.learn_weight_for_masks,
                rematerialize=args.rematerialize,
                binarize=args.binarize)

else:
    logger.info("Constructing original PixelCNN++")
    model_sample = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix,
                rematerialize=args.rematerialize)

    assert not args.randomize_order

model_sample = model_sample.cuda()

# Create optimizer
# NOTE: PixelCNN++ TF repo uses betas=(0.95, 0.9995), different than PyTorch defaults
optimizer = optim.Adam(model_sample.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

if args.amp_opt_level:
    # Enable mixed precision training
    from apex import amp
    model_sample, optimizer = amp.initialize(model_sample, optimizer, opt_level=args.amp_opt_level)

model = nn.DataParallel(model_sample)

# Load model parameters from checkpoint
if args.load_params:
    if os.path.exists(args.load_params):
        load_params = args.load_params
    else:
        load_params = os.path.join(run_dir, args.load_params)
    # Load params
    print('loading')
    checkpoint_epochs, checkpoint_step = load_part_of_model(load_params,
                                           model=model.module,
                                           optimizer=None if args.do_not_load_optimizer else optimizer)
    logger.info(f"Model parameters loaded from {load_params}, from after {checkpoint_epochs} training epochs, {checkpoint_step} training steps")
elif args.load_last_params:
    # Find the most recent checkpoint (highest epoch number).
    checkpoint_re = f"{args.exp_id}_ep([0-9]+)\\.pth"
    checkpoint_files = []
    checkpoint_epochs = []
    for f in os.listdir(run_dir):
        match = re.match(checkpoint_re, f)
        if match:
            checkpoint_files.append(f)
            ep = int(match.group(1))
            checkpoint_epochs.append(ep)
            logger.info(f"Found checkpoint {f} with {ep} epochs of training")
    if checkpoint_files:
        last_checkpoint_name = checkpoint_files[int(np.argmax(checkpoint_epochs))]
        load_params = os.path.join(run_dir, last_checkpoint_name)
        logger.info(f"Most recent checkpoint: {last_checkpoint_name}")
        # Load params
        checkpoint_epochs, checkpoint_step = load_part_of_model(load_params,
                                            model=model.module,
                                            optimizer=None if not args.do_not_load_optimizer else optimizer)
        logger.info(f"Model parameters loaded from {load_params}, from after {checkpoint_epochs} training epochs")
    else:
        logger.info("No checkpoints found")
        checkpoint_epochs = -1
        checkpoint_step = -1
else:
    checkpoint_epochs = -1
    checkpoint_step = -1

if checkpoint_epochs > 0:
    # Decrease learning rate since we resumed
    logger.info("Adjusting lr due to checkpoint resumption. Before adjustment, lr=%f", scheduler.get_last_lr()[0])
    for _ep in range(checkpoint_epochs + 1):
        scheduler.step()
    logger.info("After adjustment, lr=%f", scheduler.get_last_lr()[0])

# Initialize exponential moving average of parameters
if args.ema < 1:
    ema = EMA(args.ema)
    ema.register(model.module)

# load vqvae, used for sampling
vqvae_model = VQVAETop()
ckpt = torch.load(args.vqvae_path)
torch_devices = [0]
device = "cuda:" + str(torch_devices[0])
vqvae_model = nn.DataParallel(vqvae_model, torch_devices).to(device)
vqvae_model.load_state_dict(ckpt)
vqvae_model = vqvae_model.to('cuda')
vqvae_model.eval()

obs = (1, 32, 32)

def test(model, test_loader, epoch="N/A", progress_bar=True,
         slice_op=None, sliced_obs=obs):
    #logger.info(f"Testing with ensemble of {len(all_masks)} orderings")
    test_loss = 0.
    pbar = tqdm.tqdm(test_loader,
                     desc=f"Test after epoch {epoch}",
                     disable=(not progress_bar),
                     total=test_total)
    num_images = 0

    possible_masks = []
    for batch_idx, (full_input,_) in enumerate(pbar):
        if batch_idx < 5:
            for i in range(full_input[1].shape[0]):
                possible_masks.append(get_masks(np.array(full_input[1][i]), obs[1], obs[2], 3, 2, plot=False))

        ourdata = full_input[0].cuda(non_blocking=True)  # [-1, 1] range images

        num_images += ourdata.shape[0]

        input = (
            F.one_hot(ourdata[:,0,:,:].to(torch.int64), N_CLASS).permute(0, 3, 1, 2).to(torch.float32)
        )

        input_var = Variable(input)

        # now we calculate likelihood in ordering corresponding to each image
        # we don't average across multiple.
        masks_init = []
        masks_undilated = []
        masks_dilated = []
        # before, all_masks[index] was num_gpu,9,32*32
        # we change it to be 1,9,32*32, so that we can 
        # instead split up the entire batch across machines
        for i in range(full_input[1].shape[0]):
            mask_init, mask_undilated, mask_dilated = possible_masks[np.random.randint(0,len(possible_masks))]
            masks_init.append(mask_init[0:1])
            masks_undilated.append(mask_undilated[0:1])
            masks_dilated.append(mask_dilated[0:1])
        masks_init = torch.stack(masks_init).repeat(1, 513, 1, 1).view(-1,9,32*32).cuda(non_blocking=True)
        masks_undilated = torch.stack(masks_undilated).repeat(1, 160, 1, 1).view(-1,9,32*32).cuda(non_blocking=True)
        masks_dilated = torch.stack(masks_dilated).repeat(1, 80, 1, 1).view(-1,9,32*32).cuda(non_blocking=True)
        new_input = [input, masks_init, masks_undilated, masks_dilated]
        outputs = [model(new_input)]

        order_prefix = "_".join(args.order)

        loss = loss_op(outputs[0], ourdata[:,0].to(torch.int64))

        test_loss += loss.item()
        del loss 

        deno = num_images * np.prod(sliced_obs) * np.log(2.)
        pbar.set_description(f"Test after epoch {epoch} {test_loss / deno}")

    deno = num_images * np.prod(sliced_obs) * np.log(2.)
    assert deno > 0, embed()
    test_bpd = test_loss / deno
    return test_bpd


def get_sampling_images(loader):
    # Get batch of images to complete for inpainting, or None for --sample_region=full
    if args.sample_region == "full":
        return None
    logger.info('getting batch of images to complete...')
    # Get sample_batch_size images from test set
    batches_to_complete = []
    sample_iter = iter(loader)
    tmp = next(sample_iter)

    batch_to_complete = [tmp[0][0],tmp[0][1]]
    return batch_to_complete

def sample(model, generation_idx, mask_init, mask_undilated, mask_dilated, batch_to_complete, obs):
    batch_to_complete_full = torch.clone(batch_to_complete)
    batch_to_complete = (
        F.one_hot(batch_to_complete[:,0,:,:].to(torch.int64), N_CLASS).permute(0, 3, 1, 2).to(torch.float32)
    )
    model.eval()
    if args.sample_region == "full":
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        data = data.cuda()
        sample_idx = generation_idx
        context = None
        batch_to_complete = None
    elif args.sample_region in ['custom']:
        # here, we have background mask
        data = batch_to_complete.clone().cuda()

        # Get indices of sampling region, need to do this for each image in batch
        sample_indices = []
        for image_number in range(batch_to_complete.shape[0]):
            sample_region = set()

            # Sort according to generation_idx
            sample_idx_this = []
            num_added = 0
            for www, (i, j) in enumerate(generation_idx[image_number]):
                if www > int(32*32*.6):
                    sample_idx_this.append([i, j])
                    num_added += 1 
            sample_idx_this = np.array(sample_idx_this, dtype=np.int)

            sample_indices.append(sample_idx_this)

            data[image_number, :, sample_idx_this[:, 0], sample_idx_this[:, 1]] = 0
            context = rescaling_inv(data).cpu()
    for n_pix, (_, _) in enumerate(tqdm.tqdm(sample_indices[0], desc="Sampling pixels")):
        data_v = Variable(data)
        new_input = [data_v, mask_init, mask_undilated, mask_dilated]
        out = model(new_input, sample=True)
                
        for image_number in range(out.shape[0]):
            (i, j) = sample_indices[image_number][n_pix]
            prob = torch.softmax(out[:, :, i, j] / TEMPERATURE, 1)
            new_samples = torch.multinomial(prob, 1).squeeze(-1)
            data[image_number, :, i, j] = (
                F.one_hot(new_samples[image_number].to(torch.int64), N_CLASS).to(torch.float32)
            )

    print(loss_op(data, batch_to_complete_full[:,0].to(torch.int64).cuda()))

    if batch_to_complete is not None and context is not None:
        # Interleave along batch dimension to visualize GT images
        difference = torch.abs(data.cpu() - batch_to_complete.cpu())
        data = torch.stack([context, data.cpu(), batch_to_complete.cpu(), difference], dim=1).view(-1, *data.shape[1:])

    return data

if args.mode == "train":
    logger.info("starting training")
    writer = SummaryWriter(log_dir=run_dir)
    global_step = checkpoint_step + 1
    min_train_bpd = 1e12
    min_test_bpd_by_obs = {obs: 1e12 for obs in resized_obses}
    last_saved_epoch = -1
    for epoch in range(checkpoint_epochs + 1, args.max_epochs):
        train_loss = 0.
        time_ = time.time()
        model.train()
        possible_masks = []                
        
        for batch_idx, (full_input,_) in enumerate(tqdm.tqdm(train_loader, desc=f"Train epoch {epoch}", total=train_total)):
            # for efficiency, we don't load all masks, we sample the ones at the start of the batch.
            if batch_idx < 5:
                for i in range(full_input[1].shape[0]):
                    possible_masks.append(get_masks(np.array(full_input[1][i]), obs[1], obs[2], 3, 2, plot=False))
            ourdata = full_input[0].cuda(non_blocking=True)  # [-1, 1] range images

            obs = ourdata.shape[1:]

            input = (
                F.one_hot(ourdata[:,0,:,:].to(torch.int64), N_CLASS).permute(0, 3, 1, 2).to(torch.float32)
            )
            
            masks_init = []
            masks_undilated = []
            masks_dilated = []
            # before, all_masks[index] was num_gpu,9,32*32
            # we change it to be 1,9,32*32, so that we can 
            # instead split up the entire batch across machines
            for mask_index in range(full_input[0].shape[0]):
                mask_init, mask_undilated, mask_dilated = possible_masks[np.random.randint(0,len(possible_masks))]
                masks_init.append(mask_init[0:1])
                masks_undilated.append(mask_undilated[0:1])
                masks_dilated.append(mask_dilated[0:1])
            masks_init = torch.stack(masks_init).repeat(1, 513, 1, 1).view(-1,9,32*32).cuda(non_blocking=True)
            masks_undilated = torch.stack(masks_undilated).repeat(1, 160, 1, 1).view(-1,9,32*32).cuda(non_blocking=True)
            masks_dilated = torch.stack(masks_dilated).repeat(1, 80, 1, 1).view(-1,9,32*32).cuda(non_blocking=True)
            new_input = [input, masks_init, masks_undilated, masks_dilated]
            output = model(new_input)
            loss = loss_op(output, ourdata[:,0].to(torch.int64))
            deno = args.batch_size * np.prod(obs) * np.log(2.)
            assert deno > 0, embed()
            train_bpd = loss / deno
            if args.minimize_bpd:
                loss = train_bpd

            if batch_idx % args.accum_freq == 0:
                optimizer.zero_grad()
            if args.amp_opt_level:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (batch_idx + 1) % args.accum_freq == 0:
                if args.clip > 0:
                    # Compute and rescale gradient norm
                    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                else:
                    # Just compute the gradient norm
                    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
                    gradient_norm = 0
                    for p in parameters:
                        param_norm = p.grad.data.norm(2)
                        gradient_norm += param_norm.item() ** 2
                    gradient_norm = gradient_norm ** (1. / 2)
                writer.add_scalar('train/gradient_norm', gradient_norm, global_step)
                optimizer.step()
                if args.ema < 1:
                    ema.update(model.module)
            train_loss += loss.item()

            writer.add_scalar('train/bpd', train_bpd.item(), global_step)
            min_train_bpd = min(min_train_bpd, train_bpd.item())
            writer.add_scalar('train/min_bpd', min_train_bpd, global_step)

            if batch_idx >= 100 and train_bpd.item() >= 10:
                logger.warning("WARNING: main.py: large batch loss {} bpd".format(train_bpd.item()))

            if (batch_idx + 1) % args.print_every == 0: 
                deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
                average_bpd = train_loss / args.print_every if args.minimize_bpd else train_loss / deno
                logger.info('train bpd : {:.4f}, train loss : {:.1f}, time : {:.4f}, global step: {}'.format(
                    average_bpd,
                    train_loss / args.print_every,
                    (time.time() - time_),
                    global_step))
                train_loss = 0.
                time_ = time.time()

            if (batch_idx + 1) % args.accum_freq == 0:
                global_step += 1
            
        # decrease learning rate
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            
            save_dict = {}
            
            if epoch == 0 or epoch == 1 or (epoch + 1) % args.test_interval == 0:
                for obs in resized_obses:
                    # test with all masks
                    logger.info(f"testing with obs {obs2str(obs)}...")
                    test_bpd = test(model,
                                    test_loader_by_obs[obs],
                                    epoch,
                                    progress_bar=True)
                    writer.add_scalar(f'test/bpd_{obs2str(obs)}', test_bpd, global_step)
                    logger.info(f"test loss for obs {obs2str(obs)}: %s bpd" % test_bpd)
                    save_dict[f"test_loss_{obs2str(obs)}"] = test_bpd

                    if args.test_masks:
                        # test with held-out masks, e.g. to test generalization to other orders
                        test_limit_bpd = test(model,
                                        test_loader_by_obs[obs],
                                        epoch,
                                        progress_bar=True)
                        writer.add_scalar(f'test_limit/bpd_{obs2str(obs)}', test_limit_bpd, global_step)
                        logger.info(f"test with args.test_masks={args.test_masks} loss for obs {obs2str(obs)}: %s bpd" % test_limit_bpd)

                    # Log min test bpd for smoothness
                    min_test_bpd_by_obs[obs] = min(min_test_bpd_by_obs[obs], test_bpd)
                    writer.add_scalar(f'test/min_bpd_{obs2str(obs)}', min_test_bpd_by_obs[obs], global_step)
                    if obs == dataset_obs:
                        writer.add_scalar(f'test/bpd', test_bpd, global_step)
                        writer.add_scalar(f'test/min_bpd', min_test_bpd_by_obs[obs], global_step)

            # Save checkpoint so we have checkpoints every save_interval epochs, as well as a rolling most recent checkpoint
            save_path = os.path.join(run_dir, f"{args.exp_id}_ep{epoch}.pth")
            logger.info('saving model to %s...', save_path)
            save_dict["epoch"] = epoch
            save_dict["global_step"] = global_step
            save_dict["args"] = vars(args)
            save_dict["model_state_dict"] = model.module.state_dict()
            save_dict["optimizer_state_dict"] = optimizer.state_dict()
            if args.ema < 1:
                save_dict["ema_state_dict"] = ema.state_dict()
            torch.save(save_dict, save_path)
            if (epoch + 1) % args.save_interval != 0: 
                # Remove last off-cycle checkpoint
                remove_path = os.path.join(run_dir, f"{args.exp_id}_ep{last_saved_epoch}.pth")
                if os.path.exists(os.path.join(run_dir, f"{args.exp_id}_ep{last_saved_epoch}.pth")):
                    logger.info('deleting checkpoint at %s', remove_path)
                    os.remove(remove_path)
                last_saved_epoch = epoch
            
            if (epoch + 1) % args.sample_interval == 0 or epoch == 0: 
                for obs in resized_obses:
                    batch_to_complete = get_sampling_images(sample_loader_by_obs[obs])
                    masks_init = []
                    masks_undilated = []
                    masks_dilated = []
                    for image_num in range(batch_to_complete[0].shape[0]):
                        mask_init, mask_undilated, mask_dilated = get_masks(np.array(batch_to_complete[1][image_num]), obs[1], obs[2], 3, 2, plot=False)
                        masks_init.append(mask_init[0:1])
                        masks_undilated.append(mask_undilated[0:1])
                        masks_dilated.append(mask_dilated[0:1])
                    masks_init = torch.stack(masks_init).repeat(1, 513, 1, 1).view(-1,9,32*32).cuda(non_blocking=True)
                    masks_undilated = torch.stack(masks_undilated).repeat(1, 160, 1, 1).view(-1,9,32*32).cuda(non_blocking=True)
                    masks_dilated = torch.stack(masks_dilated).repeat(1, 80, 1, 1).view(-1,9,32*32).cuda(non_blocking=True)
                    all_masks = [masks_init, masks_undilated, masks_dilated]
                    sample_t = sample(model, batch_to_complete[1], *all_masks, batch_to_complete[0], obs)
                    sample_save_path = os.path.join(run_dir, f"tsample_obs{obs2str(obs)}_{epoch}.png")
                    # decode sample
                    
                    with torch.no_grad():
                        sample_t = torch.argmax(sample_t, dim=1)
                        sample_t_out = vqvae_model.module.decode_code(sample_t.to(torch.int64).to('cuda'))
                    utils.save_image(sample_t_out*.5+.5, sample_save_path, nrow=4, padding=5, pad_value=1, scale_each=False)
            
        if "celeba" in args.dataset:
            # Need to manually re-create loaders to reset
            train_loader, test_loader_by_obs = get_celeba_dataloaders()
