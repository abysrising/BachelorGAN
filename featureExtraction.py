from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


base_model = VGG16(weights='imagenet', include_top=False)
content_layers = ['block4_conv2']
style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
content_extractor = Model(inputs=base_model.input, outputs=[base_model.get_layer(name).output for name in content_layers])
style_extractor = Model(inputs=base_model.input, outputs=[base_model.get_layer(name).output for name in style_layers])

def extract_feature_maps_and_style_vectors(image):
    # Öffnen des Bildes als PIL-Image
    pil_image = Image.open(image)
    # Umwandeln in ein Numpy-Array
    image_array = np.array(pil_image)
    # Vorverarbeitung des Bildes
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    content_features = content_extractor.predict(image_array)
    style_features = style_extractor.predict(image_array)
    style_vectors = [np.reshape(f, (-1, f.shape[-1])) for f in style_features]
    return content_features, style_vectors

image_path = "Images/exampleoriginalsize.jpg"
content_features, style_vectors = extract_feature_maps_and_style_vectors(image_path)



def plot_feature_maps(feature_maps, title_prefix):
    # Anzahl der Feature-Maps
    num_maps = len(feature_maps)

    # Plot der Feature-Maps
    fig, axs = plt.subplots(1, num_maps, figsize=(20, 4))
    
    # Wenn nur eine Feature-Map vorhanden ist, axs ist kein Array, daher kein Zugriff mit axs[i]
    if num_maps == 1:
        axs = [axs]

    for i in range(num_maps):
        map_data = feature_maps[i]

        # Überprüfen, ob die Feature-Map eine einzeilige oder einspaltige Matrix ist
        if map_data.ndim == 1:
            # Erweitern Sie die Dimensionen, um die Feature-Map darzustellen
            map_data = np.expand_dims(map_data, axis=0)
        elif map_data.ndim == 2:
            # Wenn die Feature-Map eine einzeilige oder einspaltige Matrix ist,
            # fügen Sie eine zusätzliche Dimension hinzu, um sie darzustellen
            map_data = np.expand_dims(map_data, axis=2)

        # Greifen Sie direkt auf das entsprechende Axes-Objekt zu
        ax = axs[i]
        ax.imshow(map_data[:, :, 0], cmap='gray')  # Zeige nur den ersten Filter der Feature-Map
        ax.axis('off')
        ax.set_title(f'{title_prefix} - Map {i+1}')

    plt.show()


content_title_prefix = 'Content Feature'
style_title_prefix = 'Style Feature'

# Plot der Content Feature-Maps
plot_feature_maps(content_features, content_title_prefix)

# Plot der Style Feature-Maps
for idx, style_feature_map in enumerate(style_vectors):
    plot_feature_maps([style_feature_map], f'{style_title_prefix} {idx+1}')