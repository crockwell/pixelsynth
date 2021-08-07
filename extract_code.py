import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.vqvae2.dataset import ImageFileDataset, CodeRow
from models.vqvae2.vqvae import VQVAETop

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["DEBUG"] = "False"

def extract(loader, model, device, dataset, split):
    index = 0
    embeddings = None
    if split == 'train':
        embeddings = np.zeros([32000,32,32])
    else:
        embeddings = np.zeros([8000,32,32])
    pbar = tqdm(loader)

    num_embeddings = set()
    for img, _, filename in pbar:
        img = img.to(device)

        _, _, _, id_t, _ = model.module.encode(img)
        id_t = id_t.detach().cpu().numpy()

        for file, top in zip(filename, id_t):
            index += 1
            pbar.set_description(f'inserted: {index}')
            i = file.split('/')[-1][:-4]
            num_embeddings.add(int(i))
            embeddings[int(i)] = top           

    print('num embeddings',len(num_embeddings))
    if split == 'train':
        assert(len(num_embeddings) == 32000)
    else:
        assert(len(num_embeddings) == 8000)
    try:
        os.makedirs('models/vqvae2/embeddings/')
    except:
        pass

    np.save('models/vqvae2/embeddings/%s_%s.npy' % (dataset, split), embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--split', type=str)

    args = parser.parse_args()

    device = 'cuda'

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = ImageFileDataset(args.path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model = VQVAETop()

    torch_devices = [0,1]
    device = "cuda:" + str(torch_devices[0])
    
    from torch import nn
    model = nn.DataParallel(model, torch_devices).to(device)
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()

    extract(loader, model, device, args.dataset, args.split)