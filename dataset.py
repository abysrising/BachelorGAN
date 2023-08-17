import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import json

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        json_file = self.list_files[index]
        json_path = os.path.join(self.root_dir, json_file)

        with open(json_path, 'r') as f:
            data = json.load(f)
            
        sketch = np.array(data['sketch'])  
        real_image = np.array(data['image'])  


        augmentations = config.both_transform(image=sketch, image0=real_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        
        return input_image, target_image

