import sys
sys.path.append('Models')

from Models import train_pix2pix
from Models import config_pix2pix

from Models import train_cycleGAN
from Models import config_cycleGAN

if __name__ == "__main__":
    learn_rates_pix2pix = [0.0005, 0.0004, 0.0003, 0.0001, 0.00006]
    learn_rates_cycleGAN = [0.0002, 0.0004, 0.0005, 0.00006, 0.00004]    
    
    
    for learn_rate in learn_rates_cycleGAN:
        checkpoint_gen_sketch = "Models/outputs/cycleGAN/trained_models/gens_" + str(learn_rate) + "_" + str(config_cycleGAN.NUM_EPOCHS) + ".pth.tar"
        checkpoint_gen_real = "Models/outputs/cycleGAN/trained_models/genr_" + str(learn_rate) + "_" + str(config_cycleGAN.NUM_EPOCHS) + ".pth.tar"
        checkpoint_critic_sketch = "Models/outputs/cycleGAN/trained_models/critics_" + str(learn_rate) + "_" + str(config_cycleGAN.NUM_EPOCHS) + ".pth.tar"
        checkpoint_critic_real = "Models/outputs/cycleGAN/trained_models/criticr" + str(learn_rate) + "_" + str(config_cycleGAN.NUM_EPOCHS) + ".pth.tar"
        train_cycleGAN.main(str(learn_rate),checkpoint_gen_sketch, checkpoint_gen_real, checkpoint_critic_sketch, checkpoint_critic_real, train=True)
        
    for learn_rate in learn_rates_pix2pix:
        checkpoint_disc = "Models/outputs/pix2pix/trained_models/disc_" + str(learn_rate) + "_" + str(config_pix2pix.NUM_EPOCHS) + ".pth.tar"
        checkpoint_gen = "Models/outputs/pix2pix/trained_models/gen_" + str(learn_rate) + "_" + str(config_pix2pix.NUM_EPOCHS) + ".pth.tar"
        train_pix2pix.main(str(learn_rate), checkpoint_gen, checkpoint_disc, train=True)




