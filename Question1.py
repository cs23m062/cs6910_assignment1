import wandb
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

wandb.login()

# Load fashion_MNIST data using Keras
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
wandb.init(project="Shubhodeep_Final_CS6190_DeepLearing_Assignment1",name = "Question 1")

# Create a figure to display the sample images
plt.figure(figsize=(10, 10))

# Plot one sample image for each class
for class_index in range(len(class_names)):
    # Find the index of the first image with the current class label
    sample_index = np.where(y_train == class_index)[0][0]
    
    # Get the image and its corresponding label
    image = x_train[sample_index]
    label = class_names[y_train[sample_index]]
    
    # Plot the image
    plt.subplot(2, 5, class_index + 1)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.title(label)
    plt.axis('off')

# Log the figure to Wandb
wandb.log({"Question 1": wandb.Image(plt)})
plt.show()

#wandb.log({"Question 1": output_images})
wandb.finish()