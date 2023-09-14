import torch
from utils_cycleGAN import save_checkpoint, load_checkpoint, save_some_examples, show_generated_img, show_loss_graph
import torch.nn as nn
import torch.optim as optim
import config_cycleGAN as config
from Generators_Discriminators.generator_discriminator_cycleGAN import Generator
from Generators_Discriminators.generator_discriminator_cycleGAN import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ArtDataset
import random
import os



def train_fn(disc_S, disc_R, gen_R, gen_S, train_loader, val_loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loop = tqdm(train_loader, leave=True)
    for idx, (sketch, real) in enumerate(loop):
        real = real.to(config.DEVICE)
        sketch = sketch.to(config.DEVICE)
    
        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_sketch = gen_S(real)
            D_S_real = disc_S(sketch)
            D_S_fake = disc_S(fake_sketch.detach())
            D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real))
            D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
            D_S_loss = D_S_real_loss + D_S_fake_loss

            fake_real = gen_R(sketch)
            D_R_real = disc_R(real)
            D_R_fake = disc_R(fake_real.detach())
            D_R_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
            D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
            D_R_loss = D_R_real_loss + D_R_fake_loss

            # put it togethor
            D_loss = (D_S_loss + D_R_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators S and R
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_S_fake = disc_S(fake_sketch)
            D_R_fake = disc_R(fake_real)
            loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))
            loss_G_R = mse(D_R_fake, torch.ones_like(D_R_fake))

            # cycle loss
            cycle_real = gen_R(fake_sketch)
            cycle_sketch = gen_S(fake_real)
            cycle_real_loss = l1(real, cycle_real)
            cycle_sketch_loss = l1(sketch, cycle_sketch)


            """
            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_real = gen_R(real)
            identity_sketch = gen_S(sketch)
            identity_real_loss = l1(real, identity_real)
            identity_sketch_loss = l1(sketch, identity_sketch)
            """
            # add all together
            G_loss = (
                loss_G_R
                + loss_G_S
                + cycle_real_loss * config.LAMBDA_CYCLE
                + cycle_sketch_loss * config.LAMBDA_CYCLE
                #+ identity_sketch_loss * config.LAMBDA_IDENTITY
                #+ identity_real_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

       
        if idx % 400 == 0:
           save_some_examples(gen_R, val_loader, epoch = idx/100, folder = config.OUTPUT_DIR_EVAL)
        
        if idx % 10 == 0:
            loop.set_postfix(
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),
            )

        if idx % 200 == 0:
            config.G_LOSS_LIST.append(G_loss.item())
  


def main(learn_rate, train=True):

    disc_S = Discriminator(in_channels=3).to(config.DEVICE)
    disc_R = Discriminator(in_channels=3).to(config.DEVICE)
    gen_R = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_S = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_S.parameters()) + list(disc_R.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_R.parameters()) + list(gen_R.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    #mse = nn.MSELoss()
    mse = nn.BCEWithLogitsLoss()

    if config.LOAD_MODEL:
        user_learnrate = float(input("Input old learning rate of the model you want to load: "))


        load_checkpoint(
            config.CHECKPOINT_GEN_SKETCH,
            gen_S,
            opt_gen,
            user_learnrate,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_REAL,
            gen_R,
            opt_gen,
            user_learnrate,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_SKETCH,
            disc_S,
            opt_disc,
            user_learnrate,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_REAL,
            disc_R,
            opt_disc,
            user_learnrate,
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
            config.OUTPUT_DIR_EVAL = full_folder_path
            if not os.path.exists(config.OUTPUT_DIR_EVAL):
                os.makedirs(config.OUTPUT_DIR_EVAL)
                break
            suffix += 1
        print("Saving results to: ", config.OUTPUT_DIR_EVAL)    

        for epoch in range(config.NUM_EPOCHS):
            print ("Epoch: ", epoch)
            train_fn(disc_S, disc_R, gen_R, gen_S, train_loader, val_loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler,)
            
            if config.SAVE_MODEL:
                save_checkpoint(gen_S, opt_gen, filename=config.CHECKPOINT_GEN_SKETCH)
                save_checkpoint(gen_R, opt_gen, filename=config.CHECKPOINT_GEN_REAL)
                save_checkpoint(disc_S, opt_disc, filename=config.CHECKPOINT_CRITIC_SKETCH)
                save_checkpoint(disc_R, opt_disc, filename=config.CHECKPOINT_CRITIC_REAL)

 
        show_loss_graph(config.G_LOSS_LIST, name = "generator_loss", lr = config.LEARNING_RATE, epochs = config.NUM_EPOCHS)

    else:
        for i in range(3):
            config.VAL_INDEX = random.randint(0, len(val_loader) - 1)
            show_generated_img(gen_R, val_loader)
            
        

if __name__ == "__main__":

    print("Learning rate: ", config.LEARNING_RATE, "Epochs: ", config.NUM_EPOCHS)
    main(str(config.LEARNING_RATE), train=True)
        
