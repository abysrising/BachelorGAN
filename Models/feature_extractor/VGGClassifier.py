import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class SupervisedDNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SupervisedDNNClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)  # Ändern Sie die Ausgabedimension auf Ihre Anzahl von Klassen

    def forward(self, x):
        return self.vgg(x)


def gram_matrix(feature_map):
   
    h, w, c = feature_map.size(1), feature_map.size(2), feature_map.size(0)
    
    feature_map = feature_map.view(c, -1)
    
    gram = torch.matmul(feature_map, feature_map.t()) / (h * w * c)
    
    return gram


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

    # Verketten Sie die Gram-Matrizen zu einem Style-Vektor
    style_vector = torch.cat(gram_matrices, dim=0)

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



   

