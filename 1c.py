# As you may be able to see, there are print statements for leftovers from part b. The following script is to work right after part B's script in order to train the model.

# THE MODEL'S ACCURACY REMAINS 11% BECAUSE 10, 10, 10 FOR HIDDEN LAYERS IS TOO SMALL
# DOWNSAMPLING IS CAUSING A LOSS IN PIXELS, LESS DATA
# WEIGHTS ARE TOO SMALL AS WELL
# STACKING LAYERS OF LITTLE PERCEPTRONS DON'T HELP EITHER

#    IN SHORT THE MODEL IS VERY BAD AT DETECTING THESE IMAGES FOR THE REASONS ABOVE

# Import necessary libraries
from sklearn.datasets import fetch_openml       
from sklearn.model_selection import train_test_split  
from skimage.transform import resize           
import numpy as np                              


X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)  # Download full MNIST (70,000 samples)
y = y.astype(int)                         # Convert labels from strings to integers
X = X / 255.0                             # Normalize pixel values from [0, 255] to [0.0, 1.0]

def downsample_image(img_784):
    img_28 = img_784.reshape(28, 28)      # Vector ---> 28x28 image
    img_10 = resize(img_28, (10, 10), anti_aliasing=True)  # Resize image
    return img_10.flatten()               # Flatten 10x10 back to array of 100 values


X_downsampled = np.array([downsample_image(x) for x in X])

X_temp, X_test, y_temp, y_test = train_test_split(
    X_downsampled, y, test_size=100, stratify=y)  # Split off 100 test samples

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=100, stratify=y_temp)  # Then split 100 validation samples


weights, biases, history = train(
    X_train, y_train, X_val, y_val,
    epochs=1, batch_size=32, lr=0.1
)

test_activations, _ = forward_pass(X_test, weights, biases)  # Get model predictions
test_acc = accuracy(test_activations[-1], y_test)            # Compute accuracy
# print(f"\n🎯 Test Accuracy after 1 epoch: {test_acc * 100:.2f}%")
