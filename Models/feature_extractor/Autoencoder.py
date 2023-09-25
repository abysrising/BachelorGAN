import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from dataset_feature_extractor import ArtDataset
from PIL import Image
from tqdm import tqdm
import config_feature_extractor as config

# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), #256
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #128
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #64
            nn.Conv2d(16, 8, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2) #32
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2), #64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2), #128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2), #256
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x, layers_to_extract=None):
        feature_maps = []
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if layers_to_extract and idx in layers_to_extract:
                feature_maps.append(x)
        decoded = self.decoder(x)
        return feature_maps, decoded

def train_autoencoder(autoencoder, dataloader, num_epochs=10, learning_rate=0.001):
    autoencoder.to(config.DEVICE)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate,betas=(0.5, 0.999),)

    if config.LOAD_MODEL:
        checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        autoencoder.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    if not config.LOAD_MODEL:
        criterion = nn.MSELoss()
        for epoch in range(num_epochs):
            loop = tqdm(dataloader, leave=True)
            print(f'Epoch {epoch+1}\n-------------------------------')
            for idx, x in enumerate(loop):
                x = x.to(config.DEVICE)  # Transfer data to GPU
                optimizer.zero_grad()
                _, outputs = autoencoder(x)
                loss = criterion(outputs, x)
                loss.backward()
                optimizer.step()
                
                if idx % 1000 == 0:
                    loop.set_postfix(
                        loss=loss.item()
                    )

            if config.SAVE_MODEL:
                checkpoint = {
                    "state_dict": autoencoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, config.MODEL_PATH)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def image_to_tensor(image_path):
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    tensor_image = transform(image)
    return tensor_image.unsqueeze(0)


def gram_matrix(feature_map):
   
    h, w, c = feature_map.size(1), feature_map.size(2), feature_map.size(0)
    
    feature_map = feature_map.view(c, -1)
    
    gram = torch.matmul(feature_map, feature_map.t()) / (h * w * c)
    
    return gram

def get_FM_SV(input_image, model):
    model.eval()

    layers_to_extract = [0, 2, 5]
    input_image = input_image.to(config.DEVICE)

    feature_maps, _ = model(input_image, layers_to_extract=layers_to_extract)
    style_vector = get_style_vector(feature_maps)

    return feature_maps, style_vector


def get_style_vector(f_m):
    
    feature_maps = f_m

    # Berechnen Sie die Gram-Matrizen für die ausgewählten Schichten
    gram_matrices = [gram_matrix(fm) for fm in feature_maps]

    # Verketten Sie die Gram-Matrizen zu einem Style-Vektor
    style_vector = torch.cat(gram_matrices, dim=0)

    return style_vector

def main():
    # Load your dataset using torchvision.datasets or custom dataset

    train_dataset = ArtDataset(root_dir="data/artworks/train")
    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
    )

    model = Autoencoder()
    train_autoencoder(model, dataloader)

    return model

if __name__ == "__main__":

    model = main()

    input_path = "data/artworks/train/image/21750_artist108_style23_genre1.png"
    input_image = image_to_tensor(input_path)

    feature_maps, style_vector = get_FM_SV(input_image, model)
   


