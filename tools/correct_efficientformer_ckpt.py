import re
import sys
import torch as th


ckpt = th.load(sys.argv[1], map_location='cpu')
output = sys.argv[2]


new_state = {}
for k, v in ckpt['state_dict'].items():
    s = k.split('.')
    if re.match("backbone\.network\.[1-9]\d*\.0\.*", k):
        k = k.replace('.proj.', '.conv.')
        k = k.replace('.norm.', '.bn.')
    new_state[k] = v
ckpt['state_dict'] = new_state
th.save(ckpt, output)
