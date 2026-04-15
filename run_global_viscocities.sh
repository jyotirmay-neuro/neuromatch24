#!/bin/sh

python experiments_global.py tonic/data/local/experiments/tonic/swimmer-swim/pretrained_mlp_ppo/ pretrained_mlp_vis0.pth False
python experiments_global.py tonic/data/local/experiments/tonic/swimmer-swim/mlp_256_viscosity_0.1/ pretrained_mlp_vis1.pth False
python experiments_global.py tonic/data/local/experiments/tonic/swimmer-swim/mlp_256_viscosity_0.2/ pretrained_mlp_vis2.pth False
python experiments_global.py tonic/data/local/experiments/tonic/swimmer-swim/mlp_256_viscosity_0.4/ pretrained_mlp_vis4.pth False

python experiments_global.py tonic/data/local/experiments/tonic/swimmer-swim/pretrained_ncap_ppo/ pretrained_ncap_vis0.pth True
python experiments_global.py tonic/data/local/experiments/tonic/swimmer-swim/ncap_ppo_viscosivity_0.1/ pretrained_ncap_vis1.pth True
python experiments_global.py tonic/data/local/experiments/tonic/swimmer-swim/ncap_ppo_viscosivity_0.2/ pretrained_ncap_vis2.pth True
python experiments_global.py tonic/data/local/experiments/tonic/swimmer-swim/ncap_ppo_viscosivity_0.4/ pretrained_ncap_vis4.pth True