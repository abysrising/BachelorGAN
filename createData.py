import cv2
from datasets import load_dataset
from PIL import Image
import os
import json
import numpy as np
from matplotlib import pyplot as plt
import cropLayer
import time as t
import multiprocessing
from multiprocessing import Process
import shutil
from sklearn.model_selection import train_test_split
import random

# Change size of image to "target_size"
def resize_image(input_image, target_size):
    resized_image = input_image.resize((target_size, target_size))
    return resized_image

def count_files(folder_path):
    file_count = sum([len(files) for _, _, files in os.walk(folder_path) if files])
    return file_count


class generateData():
    def __init__(self):
        self.folder_path = "data_train"
        self.x = imageToSketch()
        self.capable_cpu = 0
        self.sleep_time = 0
        dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
        print("===========================================================")
        self.adjust_data(self.folder_path, dataset)



    def process_example(self, batch, train_data_folder):
        print(multiprocessing.current_process())
        t.sleep(self.sleep_time*2)
        self.sleep_time -= 2
        

        for index, example in batch:
        
            if example["genre"] in [4, 1]:
                image = example["image"]

                resized_sketch = resize_image((self.x.image_to_sketch(image)), 256)
                resized_image = resize_image(image, 256)
            

                sketch_json_filename = f"{index}_artist{example['artist']}_style{example['style']}_genre{example['genre']}.json"
                image_json_filename = f"{index}_artist{example['artist']}_style{example['style']}_genre{example['genre']}.json"

                output_path_sketch = "data/artworks/train/sketch/" + sketch_json_filename
                output_path_image = "data/artworks/train/image/" + image_json_filename
                resized_sketch.save(output_path_sketch)
                resized_image.save(output_path_image)
             


    def adjust_data(self, output_folder, dataset):
        print("start adjusting data")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        batch_size = 25

        start_time = t.perf_counter()

        if multiprocessing.cpu_count() > 6:
            self.capable_cpu = max(multiprocessing.cpu_count()-2, 6)-8
            print("capable cpu count: ", self.capable_cpu)
        else:
            self.capable_cpu = 2
            print("capable cpu count: ", self.capable_cpu)

        print("start processing")

        # Hier die zuletzt verarbeiteten Beispiele aus einer gespeicherten Datei laden
        saved_progress = self.load_progress(output_folder)
        if saved_progress:
            batch_id, examples_progressed = saved_progress
        else:
            batch_id = 0
            examples_progressed = 0

        self.sleep_time = self.capable_cpu
        current_batch = []
        for index, example in enumerate(dataset):
            if examples_progressed <= index < examples_progressed + self.capable_cpu * batch_size:
                # Schon verarbeitete Beispiele überspringen
                continue

            if len(current_batch) >= self.capable_cpu * batch_size:
                process_list = []
                start_time = t.perf_counter()
                for i in range(self.capable_cpu):
                    a = Process(target=self.process_example, args=(current_batch[i*batch_size:(i+1)*batch_size], output_folder))
                    process_list.append(a)

                for process in process_list:
                    process.start()

                for process in process_list:
                    process.join()

                end_time = t.perf_counter()
                print("batch#", batch_id, "finished in:", end_time - start_time, "seconds")
                current_batch = []
                batch_id += 1

                examples_progressed = batch_id * self.capable_cpu * batch_size

                # Speichern des Fortschritts
                self.save_progress(output_folder, batch_id, examples_progressed)

            current_batch.append((index, example))

    def load_progress(self, output_folder):
        progress_file = os.path.join(output_folder, "progress.txt")
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                lines = f.readlines()
                if len(lines) == 2:
                    batch_id = int(lines[0])
                    examples_progressed = int(lines[1])
                    return batch_id, examples_progressed
        return None

    def save_progress(self, output_folder, batch_id, examples_progressed):
        progress_file = os.path.join(output_folder, "progress.txt")
        with open(progress_file, "w") as f:
            f.write(str(batch_id) + "\n")
            f.write(str(examples_progressed) + "\n")




        #for index, example in enumerate(sub_dataset):
        #    self.process_example(example, output_folder, index)


    


class imageToSketch():

    def initialize_net(self):
        self.protoPath = "HED/deploy.prototxt"
        self.modelPath = "HED/hed_pretrained_bsds.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)
        cv2.dnn_registerLayer("Crop", cropLayer.CropLayer)

    # Shows an image
    def show_image(self, file_path, sketch = False):
        sk = "image"
        st = 0
        if sketch == True:
            sk = "sketch"
            st = 1
        # Extrahiere den Dateinamen aus dem Pfad und entferne die Erweiterung
        file_name = os.path.splitext(os.path.basename(file_path))[st]
     
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        array = np.array(data[sk])
        array = array.astype(np.uint8)

        pil_image = Image.fromarray(array)
        image_size = pil_image.size

        plt.title(f"{file_name} - Größe: {image_size}")
        plt.imshow(pil_image)
        plt.show()

    def canny_edge_detection(self, image_path):
        # Load the image using PIL
        image = Image.open(image_path)

        # Convert PIL.Image.Image to a Numpy Array
        image_array = np.array(image)

        # Canny Edge Detection
        canny_edge = cv2.Canny(image_array, 300, 500)

        # Autocanny
        sigma = 0.2
        median = np.median(image_array)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        auto_canny = cv2.Canny(image_array, lower, upper)

        # Plot
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(image_array)

        plt.subplot(1, 3, 2)
        plt.title("AutoCanny")
        plt.imshow(auto_canny, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Canny")
        plt.imshow(canny_edge, cmap='gray')

        plt.show()


    def HED_edge_detection(self, image, thickness):

        # Load image and extract dimensions
        img = np.array(image)
        (H, W) = img.shape[:2]

        # Mean pixel values for the ImageNet training set
    
        mean_pixel_values= np.average(img, axis = (0,1))
        blob = cv2.dnn.blobFromImage(img, thickness, size=(W, H), mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]), swapRB= False, crop=False)

        #View image after preprocessing (blob)
        blob_for_plot = np.moveaxis(blob[0,:,:,:], 0,2)

        # set the blob as the input to the network and perform a forward pass
        # to compute the edges
        self.net.setInput(blob)
        hed = self.net.forward()
        hed = hed[0,0,:,:]  #Drop the other axes 
        #hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype("uint8")  #rescale to 0-255
        inverted_hed = 255 - hed

        blob = 0,
        hed = 0,

        """
        # Plot 
        plt.subplot(1, 4, 1)
        plt.title("Original")
        plt.imshow(img)
        
        plt.subplot(1, 4, 2)
        plt.title("Preprocessed")
        plt.imshow(blob_for_plot)

        plt.subplot(1, 4, 3)
        plt.title("HED")
        plt.imshow(hed, cmap='gray')

        plt.subplot(1, 4, 4)
        plt.title("Inverted HED")
        plt.imshow(inverted_hed, cmap='gray')

        plt.show()
        """
    
        return inverted_hed

    def binarization(self, image):
        # Anwenden eines Schwellenwerts, um das Bild zu binarisieren
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary_image

    def thinning(self, image):
        # Anwenden von Morphologieoperationen, um das Bild auszudünnen
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        thinned_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        return thinned_image
    
    def remove_small_components(self, binary_image, min_size):
        # Connected Component Labeling
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

        # Filter components based on size
        filtered_image = np.zeros_like(binary_image)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_size:
                filtered_image[labels == label] = 255

        """
        # Plot
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(binary_image, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Filtered")
        plt.imshow(filtered_image, cmap='gray')

        plt.show()        
        """
        return filtered_image


    def erode_image(self, binary_image, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_image = cv2.erode(binary_image, kernel, iterations=1)

        """
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(binary_image, cmap='gray')
        
        plt.subplot(1, 2, 2)
        plt.title("Eroded")
        plt.imshow(eroded_image, cmap='gray')

        plt.show()
        """
        return eroded_image
    

    def spur_removal(self, binary_image, kernel_size):
        # Dilatation oder Closing, um die Spuren zu füllen oder zu entfernen
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)


        """
        # Plot
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(binary_image, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Cleaned")
        plt.imshow(cleaned_image, cmap='gray')

        plt.show()
        """

        return cleaned_image.astype(np.uint8)

    def binarization_and_thinning(self, image):
        binary_image = self.binarization(image.astype(np.uint8))
        thinned_image = self.thinning(binary_image)

        """
        plt.subplot(1, 3, 1)
        plt.title("HED")
        plt.imshow(image, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title("Binarized HED")
        plt.imshow(binary_image, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Thinned HED")
        plt.imshow(thinned_image, cmap='gray')

        plt.show()
        """

        return thinned_image.astype(np.uint8)


    def image_to_sketch(self, image):
        if not hasattr(self, "net"):
            self.initialize_net()

        hed_thickness = 0.7
        remove_ojects_size = 100 
        erode_kernel_size = 3
        cleaned_image_kernel_size = 5

        # HED on image
        hed_image = self.HED_edge_detection(image, hed_thickness)
        # Binarize and thin output of HED
        bin_thin_image = self.binarization_and_thinning(hed_image)
        # Remove small components
        scm_image = self.remove_small_components(bin_thin_image, remove_ojects_size)
        # Erode image
        eroded_image = self.erode_image(scm_image, erode_kernel_size)
        # Remove spurs
        cleaned_image = self.spur_removal(eroded_image, cleaned_image_kernel_size)

        t_image = cv2.cvtColor(cleaned_image, cv2.COLOR_RGB2BGR)
        image_pil = Image.fromarray(t_image)
        
        return image_pil




"""
def convert_json_to_png(output_folder, input_folder, start_index=4452):
    output_directory = output_folder
    input_directory = input_folder

    for idx, file in enumerate(sorted(os.listdir(input_directory))):
        if idx < start_index:
            continue

        f = os.path.join(input_directory, file)
        with open(f, "r") as json_file:
            data = json.load(json_file)

        array = np.array(data["image"])
        array_sketch = np.array(data["sketch"])

        array = array.astype(np.uint8)
        array_sketch = array_sketch.astype(np.uint8)

        pil_sketch = Image.fromarray(array_sketch)
        pil_image = Image.fromarray(array)

        output_path_sketch = os.path.join(output_directory, "sketch", f"{file}.sketch.png")
        output_path = os.path.join(output_directory, "image", f"{file}.png")
        os.makedirs(os.path.dirname(output_path_sketch), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        pil_image.save(output_path)
        pil_sketch.save(output_path_sketch)

        print(f"Image saved: {output_path}")
        print(f"Sketch saved: {output_path_sketch}")
"""



if __name__ == "__main__":

    #y = generateData()
    x = imageToSketch()
    
    #x.canny_edge_detection("Images/exampleoriginalsize.jpg")

    #convert_json_to_png("train", "data/artworks/train")
  
   

                
    """
    directory = "data/artworks/train"
    for i in range(5):
        
        element = random.choice(os.listdir(directory))
        f = os.path.join(directory, element)

        with open(f, "r") as json_file:
            data = json.load(json_file)

        array = np.array(data["sketch"])
        array = array.astype(np.uint8)

        x.show_image(f, sketch=False)
        x.show_image(f, sketch=True)
    """
    
