import torch 
import numpy as np 

data = np.load('data/tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']

