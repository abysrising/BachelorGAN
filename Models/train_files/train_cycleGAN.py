import torch
from Models.utils_files.utils_cycleGAN import save_checkpoint, load_checkpoint, save_some_examples, show_generated_img, show_loss_graph, show_learnrate_reduction
import torch.nn as nn
import torch.optim as optim
from Models.config_files import config_cycleGAN as config
from Generators_Discriminators.generator_discriminator_cycleGAN import Generator
from Generators_Discriminators.generator_discriminator_cycleGAN import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from Models.dataset_files.dataset_cycleGAN import ArtDataset
import random
import os



def train_fn(disc_S, disc_R, gen_R, gen_S, train_loader, val_loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch, sch_disc, sch_gen):
    loop = tqdm(train_loader, leave=True)

    scheduler_gen = sch_gen
    scheduler_disc = sch_disc

    for idx, (sketch, real) in enumerate(loop):
        real = real.to(config.DEVICE)
        sketch = sketch.to(config.DEVICE)

        # Train Generators S and R
        with torch.cuda.amp.autocast():
            fake_sketch = gen_S(real)
            fake_real = gen_R(sketch)


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

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_real = gen_R(real)
            identity_sketch = gen_S(sketch)
            identity_real_loss = l1(real, identity_real)
            identity_sketch_loss = l1(sketch, identity_sketch)
            
            # add all together
            G_loss = (
                loss_G_R
                + loss_G_S
                + cycle_real_loss * config.LAMBDA_CYCLE
                + cycle_sketch_loss * config.LAMBDA_CYCLE_SKETCH
                + identity_sketch_loss * config.LAMBDA_IDENTITY
                + identity_real_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            
            D_S_real = disc_S(sketch)
            D_S_fake = disc_S(fake_sketch.detach())
            D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real))
            D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
            D_S_loss = (D_S_real_loss + D_S_fake_loss)/2

            D_R_real = disc_R(real)
            D_R_fake = disc_R(fake_real.detach())
            D_R_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
            D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
            D_R_loss = (D_R_real_loss + D_R_fake_loss)/2

            # put it togethor
            D_loss = (D_S_loss + D_R_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()



       
        if idx % 1000 ==0:
           save_some_examples(gen_R, val_loader, epoch, idx, folder = config.OUTPUT_DIR_EVAL_REAL)
           save_some_examples(gen_S, val_loader, epoch, idx, folder = config.OUTPUT_DIR_EVAL_SKETCH)        
        
        if idx % 20 == 0:
            loop.set_postfix(
                lr_rdc = config.LR_REDUCTION,
                lr_gen = opt_gen.param_groups[0]['lr'],
                lr_disc = opt_disc.param_groups[0]['lr'],
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),
            )

        if idx == len(train_loader) - 1:
            config.G_LOSS_LIST.append(G_loss.item())
            config.D_LOSS_LIST.append(D_loss.item())
            loop.set_postfix(
                lr_rdc = config.LR_REDUCTION,
                lr_gen = opt_gen.param_groups[0]['lr'],
                lr_disc = opt_disc.param_groups[0]['lr'],
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),
            )
        if idx % 500 == 0:
            config.G_LOSS_LIST.append(G_loss.item())
            config.D_LOSS_LIST.append(D_loss.item())
  
    if epoch >=10:
        config.LR_REDUCTION = True
        scheduler_gen.step(G_loss)
        scheduler_disc.step(D_loss)
        if opt_gen.param_groups[0]["lr"] not in config.LR_LIST_GEN:
            config.LR_LIST_GEN.append((opt_gen.param_groups[0]["lr"], epoch))
        if opt_disc.param_groups[0]["lr"] not in config.LR_LIST_DISC:
            config.LR_LIST_DISC.append((opt_disc.param_groups[0]["lr"], epoch))



def main(learn_rate, checkpoint_gen_sketch = config.CHECKPOINT_GEN_SKETCH, checkpoint_gen_real = config.CHECKPOINT_GEN_REAL, checkpoint_critic_sketch = config.CHECKPOINT_CRITIC_SKETCH, checkpoint_critic_real = config.CHECKPOINT_CRITIC_REAL, train=True):

    disc_S = Discriminator(in_channels=3).to(config.DEVICE)
    disc_R = Discriminator(in_channels=3).to(config.DEVICE)
    gen_R = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_S = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)


    opt_disc = optim.Adam(
        list(disc_S.parameters()) + list(disc_R.parameters()),
        lr=float(learn_rate),
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_R.parameters()) + list(gen_S.parameters()),
        lr=float(learn_rate),
        betas=(0.5, 0.999),
    )

    schedular_disc = optim.lr_scheduler.ReduceLROnPlateau(opt_disc, "min", factor=0.1, patience=1, cooldown=0, verbose=True, min_lr=1e-6)
    schedular_gen = optim.lr_scheduler.ReduceLROnPlateau(opt_gen, "min", factor=0.1, patience=1, cooldown=0, verbose=True, min_lr=1e-6) 


    L1 = nn.L1Loss()
    mse = nn.MSELoss()
   

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

        base_folder_real = "evaluation_real_" + learn_rate + "_"
        suffix = 1
        while True:
            folder_name = base_folder_real + str(suffix).zfill(2)  # Fügt führende Nullen hinzu, falls erforderlich
            full_folder_path = os.path.join(config.OUTPUT_DIR, folder_name)
            config.OUTPUT_DIR_EVAL_REAL = full_folder_path
            if not os.path.exists(config.OUTPUT_DIR_EVAL_REAL):
                os.makedirs(config.OUTPUT_DIR_EVAL_REAL)
                break
            suffix += 1
        print("Saving results to: ", config.OUTPUT_DIR_EVAL_REAL)  

        base_folder_sketch = "evaluation_sketch_" + learn_rate + "_"
        suffix = 1
        while True:
            folder_name = base_folder_sketch + str(suffix).zfill(2)  # Fügt führende Nullen hinzu, falls erforderlich
            full_folder_path = os.path.join(config.OUTPUT_DIR, folder_name)
            config.OUTPUT_DIR_EVAL_SKETCH = full_folder_path
            if not os.path.exists(config.OUTPUT_DIR_EVAL_SKETCH):
                os.makedirs(config.OUTPUT_DIR_EVAL_SKETCH)
                break
            suffix += 1
        print("Saving results to: ", config.OUTPUT_DIR_EVAL_SKETCH)  

        for epoch in range(config.NUM_EPOCHS):
            print ("Epoch: ", epoch)
            train_fn(disc_S, disc_R, gen_R, gen_S, train_loader, val_loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch, schedular_disc, schedular_gen)
            
            if config.SAVE_MODEL:
                save_checkpoint(gen_S, opt_gen, filename=checkpoint_gen_sketch)
                save_checkpoint(gen_R, opt_gen, filename=checkpoint_gen_real)
                save_checkpoint(disc_S, opt_disc, filename=checkpoint_critic_sketch)
                save_checkpoint(disc_R, opt_disc, filename=checkpoint_critic_real)

 
        show_loss_graph(config.G_LOSS_LIST, name = "generator_loss", lr = learn_rate, epochs = config.NUM_EPOCHS)
        show_loss_graph(config.D_LOSS_LIST, name = "discriminator_loss", lr = learn_rate, epochs = config.NUM_EPOCHS)

        show_learnrate_reduction(config.LR_LIST_GEN, name = "generator_learnrate_reduction", lr = learn_rate, epochs = config.NUM_EPOCHS)
        show_learnrate_reduction(config.LR_LIST_DISC, name = "discriminator_learnrate_reduction", lr = learn_rate, epochs = config.NUM_EPOCHS)

        config.LR_LIST_DISC = []
        config.LR_LIST_GEN = []
        config.G_LOSS_LIST = []
        config.D_LOSS_LIST = []


    else:
        for i in range(3):
            config.VAL_INDEX = random.randint(0, len(val_loader) - 1)
            show_generated_img(gen_R, val_loader)
            
        

if __name__ == "__main__":

    print("Learning rate: ", config.LEARNING_RATE, "Epochs: ", config.NUM_EPOCHS)
    main(str(config.LEARNING_RATE), train=True)
        
