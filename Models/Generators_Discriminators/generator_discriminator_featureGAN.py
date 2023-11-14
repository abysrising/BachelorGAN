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
from torch.nn.utils import spectral_norm
feature_map_list = []


#region RESCode - Residual Block Code
class AttentionBasedResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, style_dim=64, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None, upsample=False):
        super(AttentionBasedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)

        # Style vector processing
        self.style_fc = nn.Linear(style_dim, planes)  # Linear layer for style vector
        self.conv_style = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
       
        self.conv_upsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)

        self.style_dim = style_dim
        self.stride = stride
        self.planes = planes
        self.upsample = upsample

    def forward(self, x, style_vector):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Style vector processing
        style_weights = self.style_fc(style_vector)
        style_weights = style_weights.view(1, self.planes, 1, 1)
        style_weights = self.conv_style(style_weights)

        out = out * style_weights

        if residual.size(1) != out.size(1):
            residual = self.conv_upsample(residual)
    
        if self.upsample == True:
            residual = F.interpolate(residual, size=out.shape[2:])

      
        out += residual
        out = self.relu(out)

        return out
#endregion

#region GENCode - Generator Code

class DMILayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.weight_a = nn.Parameter(torch.ones(1, in_channels, 1, 1)*1.01)
        self.bias_a = nn.Parameter(torch.zeros(1, in_channels, 1, 1)+0.01)
        self.weight_b = nn.Parameter(torch.ones(1, in_channels, 1, 1)*0.99)
        self.bias_b = nn.Parameter(torch.zeros(1, in_channels, 1, 1)-0.01)

    def forward(self, feature_map, mask):
        if feature_map.shape[1] > mask.shape[1]:
            channel_scale = feature_map.shape[1] // mask.shape[1]
            mask = mask.repeat(1, channel_scale, 1, 1)
        
        mask = F.interpolate(mask, size=feature_map.shape[2])
        feat_a = self.weight_a * feature_map * mask + self.bias_a
        feat_b = self.weight_b * feature_map * (1-mask) + self.bias_b
        return feat_a + feat_b

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

class FMT_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, hw=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        )
        self.ad_pool = nn.AdaptiveAvgPool2d(hw)
        self.feat_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.AdaptiveAvgPool2d(hw//8),
        )
    def rescale(self, tensor, range=(0, 1)):
        return ((tensor - tensor.min()) / (tensor.max() - tensor.min()))*(range[1]-range[0]) + range[0]

    def fmt(self, content_skt, style_skt, style_feat):
        style_feat = self.ad_pool(style_feat)

        style_skt = self.rescale(self.ad_pool(style_skt))
        content_skt = self.rescale(self.ad_pool(content_skt))

        edge_feat = style_feat * style_skt
        plain_feat = style_feat * (1-style_skt)

        edge_feat = self.feat_pool(edge_feat).repeat(1,1,8,8) 
        plain_feat = self.feat_pool(plain_feat).repeat(1,1,8,8)

        return edge_feat*content_skt + plain_feat*(1-content_skt)


    def forward(self, x, sketch, style_sketch, fm_style):
        conv_x = self.conv(x)

        stylized_x = self.fmt(sketch, style_sketch, fm_style)

        stylized_x_scaled = nn.functional.interpolate(stylized_x, size=conv_x.shape[-2:], mode='bilinear', align_corners=False)
        
        concatenated_x = torch.cat([conv_x, stylized_x_scaled], dim=1)

        return concatenated_x

class GeneratorWithDMI(nn.Module):
    def __init__(self, in_channels=1, features=64, feature_map_channels = 64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)   
        ) # 128
        
        self.down1 = AttentionBasedResidualBlock(features, features*2, stride=2, upsample=True) # 64
        self.down2 = AttentionBasedResidualBlock(features*2, features*4, stride=2, upsample=True) # 32
        self.down3 = AttentionBasedResidualBlock(features*4, features*8, stride=2, upsample=True) # 16
        self.down4 = AttentionBasedResidualBlock(features*8, features*8, stride=2, upsample=True) # 8
        self.down5 = AttentionBasedResidualBlock(features*8, features*8, stride=2, upsample=True) # 4
        self.down6 = AttentionBasedResidualBlock(features*8, features*8, stride=2, upsample=True) # 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"), nn.ReLU(),
        ) # 1
        self.up1 = GENBlock(features*8, features*8, down=False, act="relu", use_dropout=True) # 2
        self.up2 =  GENBlock(features*8*2, features*8, down=False, act="relu", use_dropout=False) # 4
        self.up3 = FMT_Layer(features*8*2, features*8) # 8
        self.up4 = GENBlock(features*8*2+feature_map_channels, features*8, down=False, act="relu", use_dropout=False ) # 16
        self.up5 = FMT_Layer(features*8*2, features*4) # 32
        self.up6 = AdaINLayer(features*4*2+feature_map_channels, features*2) # 64
        self.up7 = AdaINLayer(features*2*2, features*1) # 128
        self.dmi3 = DMILayer(features*8+feature_map_channels)
        self.dmi5 = DMILayer(features*4+feature_map_channels)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, 3, kernel_size=4, stride=2, padding=1), 
            nn.Tanh()
 
        )   # 256
       
       
        self.res1 = AttentionBasedResidualBlock(features*8, features*16, 64)
      
 


    def forward(self, x, style_maps, style_vector, y_sketch):
        
    
        d1 = self.initial_down(x)
        d2 = self.down1(d1, style_vector)
        d3 = self.down2(d2, style_vector)
        d4 = self.down3(d3, style_vector)    
        d5 = self.down4(d4, style_vector)
        d6 = self.down5(d5, style_vector)
        d7 = self.down6(d6, style_vector)

        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1), x , y_sketch, style_maps[0])
     
        dmi3 = self.dmi3(up3, x)
        up4 = self.up4(torch.cat([dmi3, d5], 1))
    
        up5 = self.up5(torch.cat([up4, d4], 1), x, y_sketch, style_maps[0])
        dmi5 = self.dmi5(up5, x)

        up6 = self.up6(torch.cat([dmi5, d3], 1), style_maps[0])
        up7 = self.up7(torch.cat([up6, d2], 1), style_maps[0])
        
        return self.final_up(torch.cat([up7, d1], 1))
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

    def __init__(self, in_channel, style_dim, content_channel=1):
        super().__init__()
        
        self.style_mu_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, in_channel, 4, 2, 1)), nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(in_channel, in_channel, 4, 2, 1)), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1))
        self.to_style = nn.Linear(in_channel*2, style_dim)

        self.to_content = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)), nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(in_channel, content_channel, 3, 1, 1)), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(256))

    def forward(self, feat):
        b, c, _, _ = feat.size()

        style_mu = self.style_mu_conv(feat)

        feat_no_style_mu = feat - style_mu
        style_sigma = feat_no_style_mu.view(b, c, -1).std(-1)
        feat_content = feat_no_style_mu / style_sigma.view(b,c,1,1)
        style = self.to_style( torch.cat([style_mu.view(b,-1), style_sigma.view(b,-1)], dim=1) )
        content = self.to_content(feat_content)
        
        return content, style

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
    
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], style_dim=64):
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
        self.idn_layer = IDNLayer(features[3], style_dim)

        """"
        self.RESBlock1 = AttentionResidualBlock(features[1], features[1])
        self.RESBlock2 = AttentionResidualBlock(features[2], features[2])
        self.RESBlock3 = AttentionResidualBlock(features[3], features[3])
        """

    def forward(self, y, style_vector):


        init = self.initial(y)
        
        down1 = self.down1(init)
        #r1 = self.RESBlock1(down1, style_vector)
        
        down2 = self.down2(down1)
        #r2 = self.RESBlock2(down2, style_vector)
       
        down3 = self.down3(down2)
        #r3 = self.RESBlock3(down3, style_vector)

        
        final = self.final(down3)
        
        
        predicted_sketch, predicted_style = self.idn_layer(down3)
      

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

def gen_test(x = None, style_maps = None, input_image = None, style_vector = None, y_sketch = None):

    x = x.cuda()
    style_maps = [style_map.cuda() for style_map in style_maps]

    y_sketch = y_sketch.cuda()
    style_vector = style_vector.cuda()

    
    model = GeneratorWithDMI(in_channels=1, features=64).cuda()
    preds = model(x, style_maps, style_vector, y_sketch)

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
    #disc_test()


    feature_extractor = train_VGGClassifier()
    input_path = "C:/Users/ls26527/GAN/BachelorGAN/data/artworks/train/image/151_artist11_style12_genre4.png"
    x_path = "C:/Users/ls26527/GAN/BachelorGAN/data/artworks/train/sketch/151_artist11_style12_genre4.sketch.png"
    
    input_image = image_to_tensor(input_path)
 
    sketch = Image.open(x_path).convert("L")
    sketch = np.array(sketch).astype(np.float32)

    augmentations = both_transform(image=sketch)
    
  
    sketch_image = augmentations["image"]
    sketch_image = transform_only_input(image=sketch_image)["image"] 
    sketch_image = sketch_image.unsqueeze(0)

    y_sketch = sketch_image


    feature_maps, style_vector = get_FM_SV_VGG(input_image, feature_extractor)
    
    gen_test(sketch_image, feature_maps, input_image, style_vector, sketch_image)
    