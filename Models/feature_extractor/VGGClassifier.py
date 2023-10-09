import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*'weights' are deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*The parameter 'pretrained' is deprecated.*")

class SupervisedDNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SupervisedDNNClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)  # Ändern Sie die Ausgabedimension auf Ihre Anzahl von Klassen

    def forward(self, x):
        return self.vgg(x)

def gram_matrix(feature_map):
    # Annahme: Die Eingabe feature_map hat die Form [1, C, H, W]
    batch_size, num_channels, height, width = feature_map.size()
    feature_map = feature_map.view(batch_size, num_channels, -1)  
    gram = torch.matmul(feature_map, feature_map.permute(0, 2, 1))  
    gram /= (num_channels * height * width) 
    return gram.unsqueeze(0)  

def get_feature_maps(input_image, model):
    model.eval()

    layers_to_extract = [0, 5, 10] 

    feature_maps = []
    x = input_image

    for idx, layer in enumerate(model.vgg.features):
        x = layer(x)
        if idx in layers_to_extract:
            feature_maps.append(x)

    return feature_maps

def get_style_vector(f_m, model):
    model.eval()

    feature_maps = f_m

    # Berechnen Sie die Gram-Matrizen für die ausgewählten Schichten
    gram_matrices = [gram_matrix(fm) for fm in feature_maps]
    style_vectors = [gram.view(1, -1, 1, 1) for gram in gram_matrices]

    # Verketten Sie die Gram-Matrizen zu einem Style-Vektor
    style_vector = torch.cat(style_vectors, dim=1)
    style_vector = style_vector.mean(dim=1, keepdim=True).expand(-1, 64, -1, -1) 
    return style_vector


def image_to_tensor(image_path):
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    tensor_image = transform(image)

    return tensor_image.unsqueeze(0)


def get_FM_SV(input_image, model):
    model.eval()
    
    feature_maps = get_feature_maps(input_image, model)
    style_vector = get_style_vector(feature_maps, model)

    return feature_maps, style_vector

def main():
    model = SupervisedDNNClassifier(2)
    model.eval()

    return model

if __name__ == "__main__":

    model = main()

    input_path = "data/artworks/train/image/21750_artist108_style23_genre1.png"
    input_image = image_to_tensor(input_path)

    feature_maps, style_vector = get_FM_SV(input_image, model)
    print(style_vector.shape)
    for fm in feature_maps:
        print(fm.shape)



   

