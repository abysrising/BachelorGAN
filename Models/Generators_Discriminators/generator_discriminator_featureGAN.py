import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("C:/Users/ls26527/GAN/BachelorGAN")
import Models.config_files.config_featureGAN as config

import random
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

from Models.feature_extractor.Autoencoder import image_to_tensor

from Models.feature_extractor.VGGClassifier import main as train_VGGClassifier
from Models.feature_extractor.Autoencoder import main as train_autoencoder

from Models.feature_extractor.Autoencoder import get_FM_SV as get_FM_SV_autoencoder
from Models.feature_extractor.VGGClassifier import get_FM_SV as get_FM_SV_VGG

feature_map_list = []


#region RESCode - Residual Block Code
class AttentionLayer(nn.Module):
    def __init__(self, in_channels, style_dim):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Linear(style_dim, in_channels)  

    def forward(self, x, style_vector):
       
        style_vector = style_vector.transpose(1, 3)
        style_weights = self.fc(style_vector)
        style_weights = style_weights.transpose(1, 3)

        return x * style_weights

class AttentionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim=64, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, padding=1)
        self.attention = AttentionLayer(out_channels, style_dim=style_dim)

    def forward(self, x, style_vector):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.attention(out, style_vector)
        out += residual

        return out
#endregion

#region GENCode - Generator Code

class DMILayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.wc = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.bc = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.wp = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.bp = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
      
    def forward(self, feature_map, sketch):
     
      
        sketch = nn.functional.interpolate(sketch, size=feature_map.shape[-2:], mode='bilinear', align_corners=False)
        sketch = sketch.expand_as(feature_map)
   
        contour_area = sketch * feature_map
        plain_area = (1 - sketch) * feature_map
    
        f0c = self.wc * contour_area + self.bc
        f0p = self.wp * plain_area + self.bp

        output_feature_map = f0c + f0p
    
        return output_feature_map


class GENBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") 
            if down 
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),

        
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class AdaINLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        )

    def adain(self, feature_map, style_map):
        #idx = int(len(feature_map_list) / 3)
        #feature_map_list.append((style_map, idx))

        style_map = nn.functional.interpolate(style_map, size=feature_map.shape[-2:], mode='bilinear', align_corners=False)
        #feature_map_list.append((style_map, idx))

        reduction_layer = nn.Conv2d(style_map.shape[1], feature_map.shape[1], kernel_size=1, bias=False)
        style_map = style_map.to(reduction_layer.weight.device, dtype=reduction_layer.weight.dtype)
        style_map = reduction_layer(style_map)
        #feature_map_list.append((style_map, idx))

        style_mean = torch.mean(style_map, dim = [2,3] , keepdim=True).cuda()
        style_std = torch.std(style_map, dim=[2,3], keepdim=True).cuda()
        feature_map_mean = torch.mean(feature_map, dim=[2,3], keepdim=True).cuda()
        feature_map_std = torch.std(feature_map, dim=[2,3], keepdim=True).cuda()

        stylized_feature_map = style_std * ((feature_map - feature_map_mean) / (feature_map_std)) + style_mean

        
        return stylized_feature_map

    def forward(self, x, style_map):
        x = self.conv(x)
   
        stylized_x = self.adain(x, style_map)

        return stylized_x

class GeneratorWithDMI(nn.Module):
    def __init__(self, in_channels=3, features=64, sketch_channels=1):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)   
    
        ) # 128
        
        self.down1 = GENBlock(features, features*2, down=True, act="leaky", use_dropout=False) # 64
        self.down2 = GENBlock(features*2, features*4, down=True, act="leaky", use_dropout=False) # 32
        self.down3 = GENBlock(features*4, features*8, down=True, act="leaky", use_dropout=False) # 16
        self.down4 = GENBlock(features*8, features*8, down=True, act="leaky", use_dropout=False) # 8
        self.down5 = GENBlock(features*8, features*8, down=True, act="leaky", use_dropout=False) # 4
        self.down6 = GENBlock(features*8, features*8, down=True, act="leaky", use_dropout=False) # 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"), nn.ReLU(),
        ) # 1
        self.up1 = GENBlock(features*8, features*8, down=False, act="relu", use_dropout=True) # 2
        self.up2 =  GENBlock(features*8*2, features*8, down=False, act="relu", use_dropout=False) # 4
        self.up3 = GENBlock(features*8*2, features*8, down=False, act="relu", use_dropout=False ) # 8
        self.up4 = GENBlock(features*8*2, features*8, down=False, act="relu", use_dropout=False) # 16
        self.up5 = GENBlock(features*8*2, features*4, down=False, act="relu", use_dropout=False) # 32
        self.up6 = AdaINLayer(features*4*2, features*2)
        self.up7 = AdaINLayer(features*2*2, features*1)
        self.dmi3 = DMILayer(features*8)
        self.dmi5 = DMILayer(features*4)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, 3, kernel_size=4, stride=2, padding=1), 
            nn.Tanh()
 
        )   # 256
       
        self.RESBlock1 = AttentionResidualBlock(features, features)
        self.RESBlock2 = AttentionResidualBlock(features*2, features*2)
        self.RESBlock3 = AttentionResidualBlock(features*4, features*4)
        self.RESBlock4 = AttentionResidualBlock(features*8, features*8)
        self.RESBlock5 = AttentionResidualBlock(features*8, features*8)
        self.RESBlock6 = AttentionResidualBlock(features*8, features*8)

        self.RESBlock1_1 = AttentionResidualBlock(features*8, features*8)
        self.RESBlock1_2 = AttentionResidualBlock(features*8, features*8)
        self.RESBlock1_3 = AttentionResidualBlock(features*8, features*8)
        self.RESBlock1_4 = AttentionResidualBlock(features*8, features*8)
        self.RESBlock1_5 = AttentionResidualBlock(features*4, features*4)
        self.RESBlock1_6 = AttentionResidualBlock(features*2, features*2)
        self.RESBlock1_7 = AttentionResidualBlock(features*1, features*1)

    def forward(self, x, style_maps, style_vector):
    
       
        d1 = self.initial_down(x)
        r1 = self.RESBlock1(d1, style_vector)
        d2 = self.down1(r1)
        r2 = self.RESBlock2(d2, style_vector)
        d3 = self.down2(r2)
        r3 = self.RESBlock3(d3, style_vector)
        d4 = self.down3(r3)
        r4 = self.RESBlock4(d4, style_vector)
        d5 = self.down4(r4)
        r5 = self.RESBlock5(d5, style_vector)
        d6 = self.down5(r5)
        r6 = self.RESBlock6(d6, style_vector)
        d7 = self.down6(r6)

        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        r1_1 = self.RESBlock1_1(up1, style_vector)
        up2 = self.up2(torch.cat([r1_1, d7], 1))
        r1_2 = self.RESBlock1_2(up2, style_vector)
        up3 = self.up3(torch.cat([r1_2, d6], 1))
        r1_3 = self.RESBlock1_3(up3, style_vector)
        dmi3 = self.dmi3(r1_3, x)
        up4 = self.up4(torch.cat([dmi3, d5], 1))
        r1_4 = self.RESBlock1_4(up4, style_vector)
        up5 = self.up5(torch.cat([r1_4, d4], 1))
        r1_5 = self.RESBlock1_5(up5, style_vector)
        dmi5 = self.dmi5(r1_5, x)
        up6 = self.up6(torch.cat([dmi5, d3], 1), style_maps[0])
        r1_6 = self.RESBlock1_6(up6, style_vector)
        up7 = self.up7(torch.cat([r1_6, d2], 1), style_maps[1])
        r1_7 = self.RESBlock1_7(up7, style_vector)
      

        return self.final_up(torch.cat([r1_7, d1], 1))
#endregion

#region DISCode - Discriminator Code

class StylePredictionMLP(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_styles):
        super().__init__()
        self.fc1 = nn.Linear(in_channels*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_styles * 1 * 1)  # Ausgabegröße anpassen

    def forward(self, style_vector):
        batch_size, channels, _, _ = style_vector.size()
        style_vector = style_vector.view(batch_size, -1)
        x = self.fc1(style_vector)
        x = nn.ReLU()(x)
        predicted_style = self.fc2(x)
        return predicted_style.view(batch_size, -1, 1, 1)  # Die Ausgabe auf die gewünschte Form umformen

class IDNLayer(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_styles):
        super(IDNLayer, self).__init__()
        self.in_channels = in_channels

        # Define layers for predicting µ_style
        self.conv_mu = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Define layers for predicting σ_style
        self.conv_sigma = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Style Prediction MLP
        self.style_prediction_mlp = StylePredictionMLP(in_channels, hidden_dim, num_styles)


        # Convolutional layers for predicting the sketch
        self.conv_sketch = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=0),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=2)  
        )


    def forward(self, feature_map):
        # Reshape µ_style and σ_style to [B, C, 1, 1]

        mu_style = self.conv_mu(feature_map)
        sigma_style = self.conv_sigma(feature_map)

        batch_size, num_channels, height, width = feature_map.size()
        mu_style = mu_style.view(batch_size, num_channels, height, width).mean(dim=(2, 3), keepdim=True)
        sigma_style = sigma_style.view(batch_size, num_channels, height, width).mean(dim=(2, 3), keepdim=True)


        # Calculate σ_style
        sigma_style = torch.sigmoid(sigma_style)

        # Style vector
        style_vector = torch.cat((mu_style, sigma_style), dim=1)

        # Style Prediction
        predicted_style = self.style_prediction_mlp(style_vector)
      
        # Calculate content feature map
        content_feature_map = (feature_map - mu_style) / (sigma_style + 1e-6)  # Small epsilon added for numerical stability

        # Calculate sketch
        predicted_sketch = self.conv_sketch(content_feature_map)
       
        return predicted_style, predicted_sketch

class DISCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect", padding = 1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], hidden_dim=128, num_styles=64):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        
        self.down1 = DISCBlock(features[0], features[1], stride=2)
        self.down2 = DISCBlock(features[1], features[2], stride=2)
        self.down3 = DISCBlock(features[2], features[3], stride=1)
        
        self.final = nn.Conv2d(features[3], 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
    
        # IDN Layer
        self.idn_layer = IDNLayer(features[3], hidden_dim, num_styles)
        self.RESBlock1 = AttentionResidualBlock(features[1], features[1])
        self.RESBlock2 = AttentionResidualBlock(features[2], features[2])
        self.RESBlock3 = AttentionResidualBlock(features[3], features[3])

    def forward(self, y, style_vector):


        init = self.initial(y)
        
        down1 = self.down1(init)
        r1 = self.RESBlock1(down1, style_vector)
        
        down2 = self.down2(r1)
        r2 = self.RESBlock2(down2, style_vector)
       
        down3 = self.down3(r2)
        r3 = self.RESBlock3(down3, style_vector)

        
        final = self.final(r3)
        
        predicted_style, predicted_sketch = self.idn_layer(r3)

        return final, predicted_style, predicted_sketch

#endregion

#region Test Code
def disc_test():
   
    model = Discriminator(in_channels=3)

    y = torch.randn((1, 3, 256, 256))
    style_vector = torch.randn((1, 64, 1, 1))

    fake, predicted_style, predicted_sketch = model(y, style_vector)

    print(fake.shape, predicted_style.shape, predicted_sketch.shape)
    bce = nn.BCEWithLogitsLoss()
    prediction = bce(fake, torch.ones_like(fake))
    print(prediction.item())
    
def visualize_feature_maps(feature_maps, title, num_feature_maps_to_show=10):
    num_feature_maps = feature_maps.shape[1]
    
    # Wählen Sie zufällige Indizes für die zu zeigenden Feature Maps aus
    random_indices = random.sample(range(num_feature_maps), num_feature_maps_to_show)
    
    # Erstellen Sie eine Abbildung mit den ausgewählten Feature Maps
    fig, axes = plt.subplots(1, num_feature_maps_to_show, figsize=(15, 3))

    for i, idx in enumerate(random_indices):
        ax = axes[i]
        ax.imshow(feature_maps[0, idx].cpu().detach().numpy(), cmap='viridis')
        ax.axis('off')
        ax.set_title(f'{idx + 1}')

    fig.suptitle(title)
    plt.show()

def gen_test(x = None, style_maps = None, input_image = None, style_vector = None):

    if x is None and style_maps is None:
        x = torch.randn(1, 1, 256, 256).cuda()
        style_maps = [torch.randn(1, 64, 256, 256).cuda(), torch.randn(1, 128, 128, 128).cuda(), torch.randn(1, 256, 64, 64).cuda()]
    else:
        x = x.cuda()
        style_maps = [style_map.cuda() for style_map in style_maps]

    style_vector = style_vector.cuda()

    model = GeneratorWithDMI(in_channels=1, features=64).cuda()
    preds = model(x, style_maps, style_vector)

    print(preds.shape)
    plt.imshow(input_image[0].permute(1, 2, 0))


    for fm, name in feature_map_list:
        visualize_feature_maps(fm, name)

both_transform = A.Compose(
    [A.Resize(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    ], 
    additional_targets={"image0": "image"},
     
)

transform_only_input = A.Compose(
    [
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

#endregion

if __name__ == "__main__":
    disc_test()

    """
    feature_extractor = train_VGGClassifier()
    input_path = "data/artworks/train/image/21750_artist108_style23_genre1.png"
    x_path = "data/artworks/train/image/21750_artist108_style23_genre1.png"
    
    input_image = image_to_tensor(input_path)
 
    sketch = Image.open(x_path).convert("L")
    sketch = np.array(sketch).astype(np.float32)

    augmentations = both_transform(image=sketch)
    
  
    sketch_image = augmentations["image"]
    sketch_image = transform_only_input(image=sketch_image)["image"] 
    sketch_image = sketch_image.unsqueeze(0)

    feature_maps, style_vector = get_FM_SV_VGG(input_image, feature_extractor)
    
    gen_test(sketch_image, feature_maps, input_image, style_vector)
    """