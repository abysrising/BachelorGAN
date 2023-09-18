import torch
import torch.nn as nn


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
    


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
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
        self.up2 = GENBlock(features*8*2, features*8, down=False, act="relu", use_dropout=True) # 4
        self.up3 = GENBlock(features*8*2, features*8, down=False, act="relu", use_dropout=True) # 8
        self.up4 = GENBlock(features*8*2, features*8, down=False, act="relu", use_dropout=False) # 16
        self.up5 = GENBlock(features*8*2, features*4, down=False, act="relu", use_dropout=False) # 32
        self.up6 = GENBlock(features*4*2, features*2, down=False, act="relu", use_dropout=False) # 64
        self.up7 = GENBlock(features*2*2, features*1, down=False, act="relu", use_dropout=False) # 128
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1), 
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))
        

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
    
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                DISCBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat([x, y], dim=1)
        else:
            # Wenn y nicht vorhanden ist, erstelle einen Null-Tensor mit der gleichen Form wie x
            y = torch.zeros_like(x)
            x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        
        return self.model(x)


def disc_test():
    model = Discriminator()

    
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    
    preds = model(x, y)
    bce = nn.BCEWithLogitsLoss()
    prediction = bce(preds, torch.ones_like(preds))
    print(preds.shape)
    print(prediction.item())


def gen_test():
    x = torch.randn(1, 3, 256, 256)
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    disc_test()
    gen_test()