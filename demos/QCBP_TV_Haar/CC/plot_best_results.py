import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

regex = re.compile(r'RES-.+-(\d)')

os.makedirs('plots', exist_ok=True)

sns.set(context='paper', style='whitegrid')
cmap = 'inferno'

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9.5,16))

with open('best_results.txt') as fd:
    for line in fd:
        exp_name = line.strip('\n')
        exp_path = os.path.join('data', exp_name)
        
        match = regex.match(exp_name)
        label = int(match.group(1))

        adv_pert = np.load(os.path.join(exp_path, 'adv_pert.npy'))
        X_rec = np.load(os.path.join(exp_path, 'im_rec.npy'))
        X_pert_rec = np.load(os.path.join(exp_path, 'im_pert_rec.npy'))
        
        # show adversarial perturbation rescaled
        sns.heatmap(
            np.abs(adv_pert), 
            xticklabels=[], yticklabels=[], 
            cmap=cmap, ax=axs[label][0])

        # show absolute difference of truth and perturbed reconstruction 
        sns.heatmap(
            np.abs(X_rec-X_pert_rec),
            xticklabels=[], yticklabels=[],
            cmap=cmap, ax=axs[label][1])

        # save perturbed and reconstruction of perturbed image
        im_rec = np.clip(np.abs(X_rec)*255,0,255).astype('uint8')
        Image.fromarray(im_rec).save(os.path.join('plots', exp_name + '-im_rec.png'))
        im_pert_rec = np.clip(np.abs(X_pert_rec)*255,0,255).astype('uint8')
        Image.fromarray(im_pert_rec).save(os.path.join('plots', exp_name + '-im_pert_rec.png'))

fig.savefig(os.path.join('plots', 'stability-subplots.png'), bbox_inches='tight', dpi=300)
