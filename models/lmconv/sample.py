import torch 
import tqdm 
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def sample(model, generation_idx, mask_init, mask_undilated, mask_dilated, batch_to_complete, obs, args, seed=0, temperature=1.0, background_mask=None):
    batch_to_complete_full = torch.clone(batch_to_complete)
    batch_to_complete = (
        F.one_hot(batch_to_complete, args.num_classes).permute(0, 3, 1, 2).to(torch.float32)
    )
        
    np.random.seed(seed)
    seed2 = seed*10 + np.random.randint(188)
    torch.manual_seed(seed2)
    model.eval()

    # here, we have background mask
    data = batch_to_complete.clone().cuda()

    # Get indices of sampling region, need to do this for each image in batch
    sample_indices = []
    for image_number in range(batch_to_complete.shape[0]):
        sample_region = set()
        
        for i in range(obs[1]):
            for j in range(obs[2]):
                if background_mask[image_number,i,j] == 1:
                    sample_region.add((i, j))

        # Sort according to generation_idx
        sample_idx_this = []
        num_added = 0
        for i, j in generation_idx[image_number]:
            if (i, j) in sample_region:
                sample_idx_this.append([i, j])
                num_added += 1 
        sample_idx_this = np.array(sample_idx_this, dtype=np.int)

        sample_indices.append(sample_idx_this)

        if len(sample_region) > 0:
            # sample region is empty if input & output 
            # image are the same, then we don't sample
            # this can happen in realestate10k
            data[image_number, :, sample_idx_this[:, 0], sample_idx_this[:, 1]] = 0

    loss_score = 0
    n_pix = 0

    myloss = nn.CrossEntropyLoss()

    for n_pix, (_, _) in enumerate(tqdm.tqdm(sample_indices[0], desc="Sampling pixels")):
        data_v = Variable(data)
        new_input = [data_v, mask_init, mask_undilated, mask_dilated]
        out = model(new_input, sample=True)
        for image_number in range(out.shape[0]):
            (i, j) = sample_indices[image_number][n_pix]
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            new_samples = torch.multinomial(prob, 1).squeeze(-1)
            for k in range(seed):
                new_samples = torch.multinomial(prob, 1).squeeze(-1)
            data[image_number, :, i, j] = (
                F.one_hot(new_samples[image_number], args.num_classes).to(torch.float32)
            )
    
    loss_score = myloss(data, batch_to_complete_full)
    # revert seeding for dataloader
    torch.manual_seed(args.dataloader_seed)
    np.random.seed(args.dataloader_seed)    

    return data, loss_score
    