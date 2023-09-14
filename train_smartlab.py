import sys
sys.path.append('Models')

from Models import train_pix2pix
from Models import config_pix2pix

from Models import train_cycleGAN
from Models import config_cycleGAN

if __name__ == "__main__":
    learn_rates_pix2pix = [0.0005, 0.0004, 0.0003, 0.0001, 0.00006,]
    learn_rates_cycleGAN = [0.00005, 0.00003, 0.00001, 0.0001, 0.0003]    

    for learn_rate in learn_rates_pix2pix:
        config_pix2pix.LEARNING_RATE = learn_rate
        train_pix2pix.main(str(config_pix2pix.LEARNING_RATE), train=True)

    for learn_rate in learn_rates_cycleGAN:
        config_cycleGAN.LEARNING_RATE = learn_rate
        train_cycleGAN.main(str(config_cycleGAN.LEARNING_RATE), train=True)
