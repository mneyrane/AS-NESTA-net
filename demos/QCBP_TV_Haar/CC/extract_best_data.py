import os
import re
from glob import glob
import numpy as np

TOP_AMOUNT = 1
NOISE_LEVEL_LABELS = (0,1,2,3)

rec_error_regex = re.compile(r'pert rec error: (.+)')
pert_size_regex = re.compile(r'pert size: (.+)')

best_exps = []

for i in NOISE_LEVEL_LABELS:
    folders = glob('data/RES-*-*-%s' % i)

    exp_error_pairs = []

    for path in folders:
        _, exp_name = os.path.split(path)
        
        with open(os.path.join(path, 'params.txt')) as fd:
            text = fd.read()

            match_re = rec_error_regex.search(text)
            match_ap = pert_size_regex.search(text)
            
            value_re = float(match_re.group(1))
            value_ap = float(match_ap.group(1))
            
            # select best based on reconstruction error
            value = value_re # ~~~ / value_ap
            
            exp_error_pairs.append((exp_name,value))

    sorted_pairs = sorted(exp_error_pairs, key=lambda x : x[1], reverse=True)
    assert sorted_pairs[0][1] >= sorted_pairs[-1][1]
    best_exps.extend([x[0] for x in sorted_pairs[:TOP_AMOUNT]])

assert len(best_exps) == len(NOISE_LEVEL_LABELS) * TOP_AMOUNT

with open('best_results.txt','w') as fd:
    for exp_name in best_exps:
        fd.write(exp_name + '\n')
