# patch wise training
import os
import torch
from HDF5Constructor import HDF5Constructor
from Dataset import CustomDataset
from DataLoader import CustomDataLoader

save_path = "/home/cglab/Desktop/Path-Graph-Convolution/h5"

patch_size = 80
num_patch = 200
seed = 990819
data_ratio = (0.90, 0.10)
batch_size = 8


def train():
    train_save_path = os.path.join(save_path, "train.h5")
    val_save_path = os.path.join(save_path, "val.h5")
    exist = True
    for path in [train_save_path, val_save_path]:
        if not os.path.exists(path):
            print("{} is not exist".format(path))
            exist = False
    if not exist:
        constructor = HDF5Constructor("/home/cglab/Desktop/Path-Graph-Convolution/noisy.exr",
                                      save_path, patch_size, num_patch, seed, data_ratio
                                      )
        constructor.construct_hdf5()
        
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    train_dataset = CustomDataset(train_save_path)
    train_num_samples = len(train_dataset)
    train_dataloader = CustomDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)
    
    val_dataset = CustomDataset(train_save_path)
    val_num_samples = len(train_dataset)
    val_dataloader = CustomDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)
    
    root_save_path = "/home/cglab/Desktop/Path-Graph-Convolution/source/model"
    

if __name__ == "__main__":
    train()