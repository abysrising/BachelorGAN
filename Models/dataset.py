import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

class ArtDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files_real = os.listdir(self.root_dir + '/image')
        self.list_files_sketch = os.listdir(self.root_dir + '/sketch')

    def __len__(self):
        return len(self.list_files_real)

    def __getitem__(self, index):
        real_image_filename = self.list_files_real[index]
        sketch_filename = self.list_files_sketch[index]

        real_image_path = os.path.join(self.root_dir, 'image', real_image_filename)
        sketch_path = os.path.join(self.root_dir, 'sketch', sketch_filename)

        real_image = np.array(Image.open(real_image_path)).astype(np.float32)
        sketch = np.array(Image.open(sketch_path)).astype(np.float32)

        augmentations = both_transform(image=sketch, image0=real_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = transform_only_input(image=input_image)["image"]
        target_image = transform_only_mask(image=target_image)["image"]

        
        return input_image, target_image

