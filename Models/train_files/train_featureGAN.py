import torch
from Models.utils_files.utils_featureGAN import save_checkpoint, load_checkpoint, save_some_examples, show_generated_img, show_loss_graph, show_learnrate_reduction
import torch.nn as nn
import torch.optim as optim
from Models.config_files import config_featureGAN as config
from Generators_Discriminators.generator_discriminator_featureGAN import Generator
from Generators_Discriminators.generator_discriminator_featureGAN import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from Models.dataset_files.dataset_featureGAN import ArtDataset
import random
import os



def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, sch_gen, sch_disc, epoch):
    loop = tqdm(loader, leave=True)

    scheduler_gen = sch_gen
    scheduler_disc = sch_disc

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)

            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())

            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))

            D_loss = ((D_real_loss + D_fake_loss) / 2) * config.LAMBDA
           

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            D_real = disc(x, y)

            G_real_loss = bce(D_real, torch.zeros_like(D_real))
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))

            G_loss = ((G_real_loss + G_fake_loss) / 2) * config.LAMBDA
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA

            G_loss = G_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


        if idx % 50 == 0:
            loop.set_postfix(
                #D_real=torch.sigmoid(D_real).mean().item(),
                #D_fake=torch.sigmoid(D_fake).mean().item(),
                #D_real_loss= D_real_loss.item(),
                #D_fake_loss= D_fake_loss.item(),
                lr_rdc = config.LR_REDUCTION,
                lr_gen = opt_gen.param_groups[0]['lr'],
                lr_disc = opt_disc.param_groups[0]['lr'],
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),
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
  
    if epoch >= 50:
        config.LR_REDUCTION = True
        scheduler_gen.step(G_loss)
        scheduler_disc.step(D_loss)
        if opt_gen.param_groups[0]["lr"] not in config.LR_LIST_GEN:
            config.LR_LIST_GEN.append((opt_gen.param_groups[0]["lr"], epoch))
        if opt_disc.param_groups[0]["lr"] not in config.LR_LIST_DISC:
            config.LR_LIST_DISC.append((opt_disc.param_groups[0]["lr"], epoch))
      

def main(learn_rate, gen_checkpoint = config.CHECKPOINT_GEN, disc_checkpoint = config.CHECKPOINT_DISC, train=True,):

    print("Learning rate: ", learn_rate, "Epochs: ", config.NUM_EPOCHS)

    random_seed = random.randint(1, 1000)
    torch.manual_seed(random_seed)
    
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)

    
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
                disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, schedular_gen, schedular_disc, epoch
            )
            
    
            if config.SAVE_MODEL and epoch % 2 == 0:
                save_checkpoint(gen, opt_gen, filename=gen_checkpoint)
                save_checkpoint(disc, opt_disc, filename=disc_checkpoint)

            if epoch % 5 == 0:
                save_some_examples(gen, val_loader, epoch, full_folder_path)



        show_loss_graph(config.G_LOSS_LIST, name = "generator_loss", lr = learn_rate, epochs = config.NUM_EPOCHS)
        show_loss_graph(config.D_LOSS_LIST, name = "discriminator_loss", lr = learn_rate, epochs = config.NUM_EPOCHS)

        show_learnrate_reduction(config.LR_LIST_GEN, name = "generator_learnrate_reduction", lr = learn_rate, epochs = config.NUM_EPOCHS)
        show_learnrate_reduction(config.LR_LIST_DISC, name = "discriminator_learnrate_reduction", lr = learn_rate, epochs = config.NUM_EPOCHS)


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