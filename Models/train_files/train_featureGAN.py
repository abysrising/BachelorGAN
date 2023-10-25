import sys
sys.path.append("C:/Users/ls26527/GAN/BachelorGAN")

import Models.config_files.config_featureGAN as config
import torch
from Models.utils_files.utils_featureGAN import save_checkpoint, load_checkpoint, save_some_examples, show_generated_img, show_loss_graph, show_learnrate_reduction
import torch.nn as nn
import torch.optim as optim
from Models.Generators_Discriminators.generator_discriminator_featureGAN import GeneratorWithDMI as Generator
from Models.Generators_Discriminators.generator_discriminator_featureGAN import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from Models.dataset_files.dataset_featureGAN import ArtDataset
import random
import os
import torch.nn.functional as F


from Models.feature_extractor.Autoencoder import image_to_tensor

from Models.feature_extractor.VGGClassifier import main as train_VGGClassifier
from Models.feature_extractor.Autoencoder import main as train_autoencoder

from Models.feature_extractor.Autoencoder import get_FM_SV as get_FM_SV_autoencoder
from Models.feature_extractor.VGGClassifier import get_FM_SV as get_FM_SV_VGG



def get_random_data(dataset, paired=True, used_indices=[]):
    if paired:
        # Verwenden Sie eine Liste, um bereits ausgewählte Indizes zu speichern
        batch_x = []
        batch_y = []
        batch_y_sketch = []

        while True:
            random_idx = random.randint(0, len(dataset) - 1)
            
            # Stellen Sie sicher, dass der Index nicht bereits verwendet wurde
            if random_idx not in used_indices:
                used_indices.append(random_idx)
                x, y = dataset[random_idx]
                batch_x.append(x)
                batch_y.append(y)
                batch_y_sketch.append(x)
                return torch.stack(batch_x), torch.stack(batch_y), torch.stack(batch_y_sketch)
    else:
        # Wählen Sie zufällige ungepaarte Elemente aus dem Dataset
        random_idx_x = random.randint(0, len(dataset) - 1)

        batch_x = []
        batch_y = []
        batch_y_sketch = []
        # Verwenden Sie eine Liste, um bereits ausgewählte x-Indizes zu speichern

        while True:
            random_idx_y = random.randint(0, len(dataset) - 1)
            
            # Stellen Sie sicher, dass der Index für y nicht bereits verwendet wurde
            if random_idx_y not in used_indices:
                break
            
        while True:
            # Stellen Sie sicher, dass x und y nicht dasselbe Element sind
            if random_idx_x != random_idx_y:
                break
            
            # Falls x und y dasselbe Element sind, wählen Sie ein neues x aus
            random_idx_x = random.randint(0, len(dataset) - 1)
        
        x = dataset[random_idx_x][0]
        y = dataset[random_idx_y][1]
        y_sketch = dataset[random_idx_y][0]

        batch_x.append(x)
        batch_y.append(y)
        batch_y_sketch.append(y_sketch)
        # Fügen Sie die ausgewählten Indizes zur entsprechenden Liste hinzu
        used_indices.append(random_idx_x)
        
        return torch.stack(batch_x), torch.stack(batch_y), torch.stack(batch_y_sketch)

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, sch_gen, sch_disc, epoch, feature_extractor, paired=True, used_indices=[]):
    loop = tqdm(loader, leave=True)

    scheduler_gen = sch_gen
    scheduler_disc = sch_disc

   
    for idx, (x, y) in enumerate(loop):
        x_gen, y_gen, y_gen_sketch = get_random_data(loader.dataset, paired, used_indices)
        
        x_gen = x_gen.to(config.DEVICE)
        y_gen = y_gen.to(config.DEVICE)
        y_gen_sketch = y_gen_sketch.to(config.DEVICE)
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)


        feature_maps, style_vector = get_FM_SV_VGG(y, feature_extractor)
        feature_maps = [fm.to(config.DEVICE) for fm in feature_maps]
        style_vector = style_vector.to(config.DEVICE)

        feature_maps_gen, style_vector_gen = get_FM_SV_VGG(y_gen, feature_extractor)
        feature_maps_gen = [fm.to(config.DEVICE) for fm in feature_maps_gen]
        style_vector_gen = style_vector_gen.to(config.DEVICE)
        
        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x_gen, feature_maps_gen, style_vector_gen, y_gen_sketch)

            D_real, predicted_style_real, predicted_sketch_real = disc(y, style_vector)
            D_fake, predicted_style_fake, predicted_sketch_fake = disc(y_fake.detach(), style_vector_gen)

            D_Content_loss = F.mse_loss(predicted_sketch_real, x)
            D_Style_loss = F.mse_loss(predicted_style_real, style_vector)

            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))

            D_loss = ((D_real_loss + D_fake_loss) / 2) 
           
            D_loss = D_loss + D_Content_loss + D_Style_loss

        disc.zero_grad()
        d_scaler.scale(D_loss).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            
            D_fake, predicted_style_fake, predicted_sketch_fake = disc(y_fake, style_vector_gen)
            
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))

            G_Content_loss = F.mse_loss(predicted_sketch_fake, x_gen) *3
            G_Style_loss = F.mse_loss(predicted_style_fake, style_vector_gen) *3

            G_loss = G_fake_loss + G_Content_loss + G_Style_loss
            if paired == False:
                additional_loss = F.mse_loss(y_fake, y_gen)
                G_loss = G_loss + additional_loss
        

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


        if idx % 50 == 0:
            loop.set_postfix(
                lr_rdc = config.LR_REDUCTION,
                lr_gen = opt_gen.param_groups[0]['lr'],
                lr_disc = opt_disc.param_groups[0]['lr'],
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),
                paired = paired,
            )

        if idx == len(loader) - 1:
            config.G_LOSS_LIST.append(G_loss.item())
            config.D_LOSS_LIST.append(D_loss.item())
            loop.set_postfix(
                #D_real_loss= D_real_loss.item(),
                #D_fake_loss= D_fake_loss.item(),
                lr_rdc = config.LR_REDUCTION,
                lr_gen = opt_gen.param_groups[0]['lr'],
                lr_disc = opt_disc.param_groups[0]['lr'],
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),

            )

    if epoch >= 5:
        config.LR_REDUCTION = True
        scheduler_gen.step(G_loss)
        scheduler_disc.step(D_loss)
        if opt_gen.param_groups[0]["lr"] not in config.LR_LIST_GEN:
            config.LR_LIST_GEN.append((opt_gen.param_groups[0]["lr"], epoch))
        if opt_disc.param_groups[0]["lr"] not in config.LR_LIST_DISC:
            config.LR_LIST_DISC.append((opt_disc.param_groups[0]["lr"], epoch))

def main(learn_rate, gen_checkpoint = config.CHECKPOINT_GEN, disc_checkpoint = config.CHECKPOINT_DISC, train=True,):

    feature_extractor_auto = train_autoencoder()
    feature_extractor_VGG = train_VGGClassifier().to(config.DEVICE)

    print("Learning rate: ", learn_rate, "Epochs: ", config.NUM_EPOCHS)

    random_seed = random.randint(1, 1000)
    torch.manual_seed(random_seed)
    
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=1).to(config.DEVICE)

    
    opt_disc = optim.Adam(disc.parameters(), lr=float(learn_rate), betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=float(learn_rate), betas=(0.5, 0.999),)


    schedular_disc = optim.lr_scheduler.ReduceLROnPlateau(opt_disc, "min", factor=0.1, patience=3, cooldown=1, verbose=True, min_lr=1e-6)
    schedular_gen = optim.lr_scheduler.ReduceLROnPlateau(opt_gen, "min", factor=0.1, patience=3, cooldown=1, verbose=True, min_lr=1e-6) 


    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        user_learnrate = float(input("Input old learning rate of the model you want to load: "))


        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, user_learnrate,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, user_learnrate,
        )
    
    

    train_dataset = ArtDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    val_dataset = ArtDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    config.RANDOM_INDEX = random.randint(0, len(val_loader) - 1)

    if train:

        base_folder = "evaluation" + learn_rate + "_"
        suffix = 1
        while True:
            folder_name = base_folder + str(suffix).zfill(2)  # Fügt führende Nullen hinzu, falls erforderlich
            full_folder_path = os.path.join(config.OUTPUT_DIR, folder_name)
            if not os.path.exists(full_folder_path):
                os.makedirs(full_folder_path)
                break
            suffix += 1
        print("Saving results to: ", full_folder_path)    

        for epoch in range(config.NUM_EPOCHS):
            print ("Epoch: ", epoch)
            train_fn(
                disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, schedular_gen, schedular_disc, epoch, feature_extractor_VGG, paired=True if epoch %2 == 0 else False, used_indices=[]
            )


            if config.SAVE_MODEL and epoch % 2 == 0:
                save_checkpoint(gen, opt_gen, filename=gen_checkpoint)
                save_checkpoint(disc, opt_disc, filename=disc_checkpoint)

            if epoch % 1 == 0 or epoch == config.NUM_EPOCHS - 1:
                save_some_examples(gen, val_loader, epoch, full_folder_path, feature_extractor_VGG, get_FM_SV_VGG)




        show_loss_graph(config.G_LOSS_LIST, name = "generator_loss", lr = learn_rate, epochs = config.NUM_EPOCHS)
        show_loss_graph(config.D_LOSS_LIST, name = "discriminator_loss", lr = learn_rate, epochs = config.NUM_EPOCHS)

        #show_learnrate_reduction(config.LR_LIST_GEN, name = "generator_learnrate_reduction", lr = learn_rate, epochs = config.NUM_EPOCHS)
        #show_learnrate_reduction(config.LR_LIST_DISC, name = "discriminator_learnrate_reduction", lr = learn_rate, epochs = config.NUM_EPOCHS)


        config.LR_LIST_GEN = []
        config.LR_LIST_DISC = []
        config.G_LOSS_LIST = []
        config.D_LOSS_LIST = []

    else:
        for i in range(3):
            config.VAL_INDEX = random.randint(0, len(val_loader) - 1)
            show_generated_img(gen, val_loader)
            
    
if __name__ == "__main__":
    main(str(config.LEARNING_RATE), train=True)

    """
    feature_extractor_auto = train_autoencoder()
    feature_extractor_VGG = train_VGGClassifier()
    input_path = "data/artworks/train/image/21750_artist108_style23_genre1.png"
    input_image = image_to_tensor(input_path)

    feature_maps, style_vector = get_FM_SV_VGG(input_image, feature_extractor_VGG)
    print(style_vector) 
    """