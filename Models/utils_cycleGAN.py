import torch
import config_cycleGAN as config
from torchvision.utils import save_image
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
import pandas as pd


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def save_some_examples(gen, val_loader, epoch, folder):
    
    for i, (x, y) in enumerate(val_loader):
        if i == config.RANDOM_INDEX:
            break
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)

        # remove normalization
        y_fake = y_fake * 0.5 + 0.5  
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5

        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x , folder + f"/input_{epoch}.png") 
        if epoch == 0:
            save_image(y, folder + f"/label_{epoch}.png") 
    gen.train()

def calculate_average_loss_change(loss_list):
    loss_changes = [loss_list[i] - loss_list[i - 1] for i in range(1, len(loss_list))]
    average_loss_change = sum(loss_changes) / len(loss_changes)
    return average_loss_change

def find_peaks(loss_list):
    loss_changes = [loss_list[i] - loss_list[i - 1] for i in range(1, len(loss_list))]
    threshold = np.std(loss_changes) * 1  # Anpassen des Schwellenwerts für große Veränderungen
    peaks = []
    for i in range(1, len(loss_changes) - 1):
        if abs(loss_changes[i]) > threshold:
            peaks.append(i + 1)
    return peaks

def show_loss_graph(loss_list, name = "loss_graph", lr = 0.0002, epochs = 200):
    x_values = list(range(1, len(loss_list) + 1))
    
    df = pd.DataFrame({"Epoch": x_values, "Loss": loss_list})
    
    sns.set(style="whitegrid")
    
    g = sns.FacetGrid(df, height=6, aspect=1.5)
    
    g.map(plt.plot, "Epoch", "Loss", color='b', label="Loss")
    
    average_change = calculate_average_loss_change(loss_list)
    
    average_change_first_30 = calculate_average_loss_change(loss_list[:30])
    
    avg_text = f"Total Avg Loss Change: {average_change:.2f}\nFirst 30 Epochs Avg Loss Change: {average_change_first_30:.2f}"
    
    first_loss = loss_list[0]
    last_loss = loss_list[-1]
    
    first_last_text = f"First Loss: {first_loss:.2f}\nLast Loss: {last_loss:.2f}"
    
    g.ax.annotate(avg_text, xy=(0.7, 0.8), xycoords='axes fraction', color='r')
    g.ax.annotate(first_last_text, xy=(0.7, 0.6), xycoords='axes fraction', color='g')
    
    g.add_legend(loc="upper left")

    # Markieren von Spitzen im Graphen
    peaks = find_peaks(loss_list)
    peak_values = [loss_list[i - 1] for i in peaks]
    
    for i, peak in enumerate(peaks):
        if loss_list[peak] > loss_list[peak - 1]:
            color = 'r'  # Rot für positive Änderungen
        else:
            color = 'g'  # Grün für negative Änderungen
        g.ax.scatter(peak, peak_values[i], c=color, marker='^', label='Peaks')

    save_name = f"{name}_lr_{lr}_epochs_{epochs}.png"
    full_folder_patch = os.path.join(config.OUTPUT_DIR, "loss_graphs")
    save_path = os.path.join(full_folder_patch, save_name)
    plt.savefig(save_path)
    
    plt.show()


def show_generated_img(gen, val_loader):

    index = config.VAL_INDEX

    folder = "validation_images/"
    full_folder_path = os.path.join(config.OUTPUT_DIR, folder)

    if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)
    
    for i, (x, y) in enumerate(val_loader):
        if i == config.VAL_INDEX:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            break
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)

        # remove normalization#
        y_fake = y_fake * 0.5 + 0.5  
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        

        save_image(y_fake, full_folder_path + f"/y_gen_{index}.png")
        save_image(x , full_folder_path + f"/input_{index}.png") 
        save_image(y, full_folder_path + f"/label_{index}.png") 

    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr