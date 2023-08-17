import concurrent.futures
import cv2
from datasets import load_dataset
from PIL import Image
import os
import json
import numpy as np
from matplotlib import pyplot as plt
import cropLayer
from sklearn.model_selection import train_test_split


# Change size of image to "target_size"
def resize_image(input_image, target_size):
    resized_image = input_image.resize((target_size, target_size))
    return resized_image

class generateData():
    def __init__(self):
        self.batch_size = 20
        self.folder_path = "data_train"
        dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
        self.adjust_data(self.folder_path, dataset)
    

    def count_files(self, folder_path):
        file_count = sum([len(files) for _, _, files in os.walk(folder_path) if files])
        return file_count

    def process_batch(self, batch, x, train_data_folder, batchid):
        index = -1
        for example in batch:
            index += 1
            if example["genre"] in [4, 1]:
            
                image = example["image"]

                resized_sketch = resize_image((x.image_to_sketch(image)), 256)
                resized_sketch_array = np.array(resized_sketch)
        

                resized_image = resize_image(image, 256)
                resized_image_array = np.array(resized_image)

                image_data = {
                    "image": resized_image_array.tolist(),
                    "sketch": resized_sketch_array.tolist(),
                    "artist": example["artist"],
                    "style": example["style"],
                    "genre": example["genre"],
                }

                json_filename = f"{index + batchid*self.batch_size}_artist{example['artist']}_style{example['style']}_genre{example['genre']}.json"
                json_filepath = os.path.join(train_data_folder, json_filename)

                with open(json_filepath, "w") as json_file:
                    json.dump(image_data, json_file)

                print(json_filepath)
                print("===========================================================")
                file_count = self.count_files(self.folder_path)
                print(f"Anzahl der Dateien im Ordner {self.folder_path}: {file_count}")
                print("===========================================================")


    def adjust_data(self, output_folder, dataset):
        train_data_folder = output_folder

        if not os.path.exists(train_data_folder):
            os.makedirs(train_data_folder)

        x = imageToSketch()
       
        batch = []
        futures = []
        batchid = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for example in dataset:
                batch.append(example)
                if len(batch) >= self.batch_size:
                    futures.append(executor.submit(self.process_batch, batch, x, train_data_folder, batchid))
                    batch = []
                    print(f"Tasks: {len(futures)}")
                    print("===========================================================")
                    batchid += 1

            # Überprüfen, ob es noch Aufgaben im letzten Batch gibt
            if len(batch) > 0:
                futures.append(executor.submit(self.process_batch, batch, x, train_data_folder, batchid))

            # Warten, bis alle Future-Objekte abgeschlossen sind
            concurrent.futures.wait(futures)

  


class imageToSketch():

    def __init__(self):
        self.protoPath = "HED/deploy.prototxt" # path to the prototxt file
        self.modelPath = "HED/hed_pretrained_bsds.caffemodel" # path to the pre-trained model
        # Load model and create blob
        self.net = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)
        # register our crop layer with the model
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

    def canny_edge_detection(self, image):

        image_array = self.convert_to_image(image)

        # Konvertiere PIL.Image.Image in ein Numpy-Array
        image_array = image_array[1]
        
        # Canny
        canny_edge = cv2.Canny(image_array, 50, 150)

        # Autocanny
        sigma = 0.3
        median = np.median(image_array)

        # Berechne die Schwellenwerte für die automatische Canny-Kantenerkennung
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




if __name__ == "__main__":

    y = generateData()
    #x = imageToSketch()
    
    """
    i = 0
    directory = "data_train"
    for element in os.listdir(directory):
        if i >= 10:
            break  
        f = os.path.join(directory, element)
        x.show_image(f, sketch=False)
        x.show_image(f, sketch=True)
        i += 1
    """


    """"
    # Pfade zu den Ordnern
    source_folder = "data_train"
    train_folder = "data/artworks/train"
    val_folder = "data/artworks/val"

    # Liste der Dateinamen in deinem Quellordner
    file_names = os.listdir(source_folder)

    # Aufteilung in Trainings- und Validierungsdaten
    train_file_names, val_file_names = train_test_split(file_names, test_size=0.3, random_state=42)

    # Verschieben der Trainingsdaten
    for file_name in train_file_names:
        src_path = os.path.join(source_folder, file_name)
        dst_path = os.path.join(train_folder, file_name)
        shutil.move(src_path, dst_path)

    # Verschieben der Validierungsdaten
    for file_name in val_file_names:
        src_path = os.path.join(source_folder, file_name)
        dst_path = os.path.join(val_folder, file_name)
        shutil.move(src_path, dst_path)
    """
  