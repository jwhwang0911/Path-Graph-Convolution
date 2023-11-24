import random
import h5py
import os
from multiprocessing import Queue, Process, Value
from sklearn.neighbors import BallTree
import multiprocessing
import numpy as np
from scipy import ndimage
import torch
import exr
from shutil import copyfile
from random import randint

def get_cropped_patches(exr_path, gt_path, patch_size, num_patches):
    data = preprocessing(exr_path, gt_path)
    patches = importance_sampling(data, patch_size, num_patches)
    cropped = list(crop(data, index, tuple(position), patch_size) for index, position in enumerate(patches))
    return cropped, patches

def crop(data, index, position, patch_size):
    print(index)
    
    half_patch = patch_size // 2
    hx, hy = half_patch, half_patch
    px, py = position
    temp = {}
    for key, value in data.items():
        if key in ['albedo', 'depth', 'normal']:
            continue
        else:
            temp[key] = value[(py-hy):(py+hy+patch_size%2), (px-hx):(px+hx+patch_size%2), :].reshape((patch_size * patch_size,-1))
        
        if key in ["noisy","patch"]:
            X = temp[key]
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            n_neighbors = 8
            ball_tree = BallTree(X).to(device)
            
            _, temp["adj_{}".format(key)] = ball_tree.query(X, k=n_neighbors + 1).to("cpu")
        
    return temp

def postprocess_log(data):
    return np.exp(data) - 1

def clip_numpy(target):
    target = np.nan_to_num(target)
    return np.clip(target,0.0, np.max(target))

def depth_process(depth):
    max_depth = np.max(depth)
    if max_depth != 0:
        depth /= max_depth
    return depth       

def preprocessing(nsy_path:torch.Tensor, gt_path:torch.Tensor):
    data = {"noisy" : None, "aux" : None, "gt":None,'normal':None, 'depth':None, 'albedo':None, 'patch': None}
    d = exr.read_all(nsy_path)
    gt= exr.read(gt_path)
    data["noisy"] = clip_numpy(d["noisy"])
    data["gt"] = gt
    data['normal'] = clip_numpy(d["normal"])
    data['depth'] = depth_process(clip_numpy(d["depth"]))
    data['albedo'] = clip_numpy(d["albedo"])
    # data['grad'] = gradient(d["default"])
    data["aux"] = np.concatenate(
        (
            data['normal'].copy(),
            data['depth'].copy(),
            data['albedo'].copy()         
        ), axis=2
    )
    
    patch_array = np.pad(data["noisy"], ((2,2),(2,2),(0,0)), mode="edge")
            
    result_list = []
    for i in range(2, patch_array.shape[0]-2):
        result_x = []
        for j in range(2, patch_array.shape[1]-2):
            window = patch_array[i-2:i+3, j-2:j+3, :].reshape(75,)
            result_x.append(window)
        result_list.append(result_x)
    result_list = np.array(result_list)
    
    data["patch"] = result_list
        
    return data

def importance_sampling(data, patch_size, num_patches):
    buffers = []
    for b in ['noisy','normal']:
        buffers.append(data[b][:,:,:])
        # data[b] 의 size는 (720,1280,3)
            
    metrics = ['relative','variance']
    weights = [1.0, 1.0]
    importance_map = get_importance_map(buffers, metrics, weights, patch_size)
    
    patches = sample_patches_dart_throwing(buffers[0].shape[:2], patch_size, num_patches)
    # 거리가 서로 2r 만큼 떨어져있는 (r은 최소 반지름) sample들을 dart throwing을 통해 잡기
    # (400 , 2)
    pad = patch_size // 2

    pruned = np.maximum(0, prune_patches(buffers[0].shape[:2], patches + pad, patch_size, importance_map) - pad)
    
    return pruned + pad

def prune_patches(exr_shapes, patches, patch_size, importance_map):
    pruned = np.empty_like(patches)
    remain = np.copy(patches)
    count, error = 0, 0
    for region in get_region_list(exr_shapes, 4*patch_size):
        current, remain = split_patches(remain, region)
        for i in range(current.shape[0]):
            x, y = current[i, 0], current[i, 1]
            if importance_map[y, x] - error > random.random():
                pruned[count, :] = [x, y]
                count += 1
                error += 1 - importance_map[y, x]
            else:
                error += 0 - importance_map[y, x]
    return pruned[:count, :]


def get_region_list(exr_shapes, step):
    regions = []
    for y in range(0, exr_shapes[0], step):
        if y//step % 2 == 0:
            xrange = range(0, exr_shapes[1], step)
        else:
            xrange = reversed(range(0, exr_shapes[1], step))
        for x in xrange:
            regions.append((x, x+step, y, y+step))
    return regions

def split_patches(patches, region):
    current = np.empty_like(patches)
    remain = np.empty_like(patches)
    current_count, remain_count = 0, 0
    for i in range(patches.shape[0]):
        x, y = patches[i, 0], patches[i, 1]
        if region[0] <= x <= region[1] and region[2] <= y <= region[3]:
            current[current_count, :] = [x, y]
            current_count += 1
        else:
            remain[remain_count, :] = [x, y]
            remain_count += 1
    return current[:current_count, :], remain[:remain_count, :]

def sample_patches_dart_throwing(exr_shapes, patch_size, num_patches, max_iter=5000):
    full_area = float(exr_shapes[0] * exr_shapes[1])
    sample_area = full_area / num_patches
    
    radius = np.sqrt(sample_area / np.pi)
    min_square_distance = (2 * radius) **2
    
    rate = 0.96
    patches = np.zeros((num_patches,2), dtype=int)
   
    x_min, x_max = 0, exr_shapes[1] - patch_size - 1
    y_min, y_max = 0, exr_shapes[0] - patch_size - 1
    
    for patch_index in range(num_patches):
        done = False
        while not done:
            for i in range(max_iter):
                x = randint(x_min, x_max)
                y = randint(y_min, y_max)
                square_distance = get_square_distance(x, y, patches[:patch_index, :])
                if square_distance > min_square_distance:
                    patches[patch_index, :] = [x, y]
                    done = True
                    break
            if not done:
                radius *= rate
                min_square_distance = (2 * radius) ** 2
    return patches

def get_square_distance(x,y,patches):
    if len(patches) == 0 :
        return np.infty
    dist = patches - [x,y]
    
    return np.sum(dist**2, axis = 1).min()
    
    

def get_importance_map(buffers, metrics, weights, patch_size):
    if len(metrics) != len(buffers):
        metrics = [metrics[0]] * len(buffers)
    
    if len(weights) != len(buffers):
        weights = [weights[0]] * len(buffers)
    
    importance_map = None
    
    for buffer, metric, weight in zip(buffers, metrics, weights):
        if metric == 'variance':
            temp = get_variance_map(buffer, patch_size, relative=False)
        elif metric == 'relative':
            temp = get_variance_map(buffer, patch_size, relative=True)
        else:
            raise ValueError('Unknown metric %s' % metric)
        
        if importance_map is None:
            importance_map = temp * weight
        else : 
            importance_map = temp * weight
    return importance_map / np.max(importance_map)

def get_variance_map(buffer, patch_size ,relative):
    mean = ndimage.uniform_filter(buffer, size = (patch_size, patch_size, 1))
    square_mean = ndimage.uniform_filter(buffer*2, size = (patch_size, patch_size,1))
    variance = np.maximum(square_mean - mean **2, 0)
    if relative:
        variance = variance / np.maximum(mean**2, 1e-4)
    
    # take the max variance along the three channels, gamma correct it to get a
    # less peaky map, and normalize it to the range [0,1]
    variance = variance.max(axis = 2)
    variance = np.minimum(variance ** (1.0 / 2.2), 1.0)
    
    return variance / np.maximum(variance.max(), 1e-4)    