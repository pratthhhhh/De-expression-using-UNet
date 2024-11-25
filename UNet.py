import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
print(tf.__version__)

import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import keras

from keras.utils import to_categorical
from keras.utils import normalize
from keras.metrics import MeanIoU
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import layers, Model

SIZE_X = 128
SIZE_Y = 128
n_classes = 83 #Number of classes for segmentation

directory_path = "./carData/train"
# Get a sorted list of image paths
img_paths = sorted(glob.glob(os.path.join(directory_path, "*.jpg")))

train_images = []

# Iterate through the sorted list of image paths
for img_path in img_paths:
    img = cv2.imread(img_path, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    # img = img.astype(np.float32)
    img = img/255.0
    train_images.append(img)
    # # print(img_path)
    # train_images.append(img)
    # #train_labels.append(label)

#Convert list to array for machine learning processing
train_images = np.array(train_images)

directory_path = "./carData/y_mask"
# Get a sorted list of image paths
img_paths = sorted(glob.glob(os.path.join(directory_path, "*.jpg")))

train_masks = []

# Iterate through the sorted list of image paths
for mask_path in img_paths:
  mask = cv2.imread(mask_path, 0)
  # mask = mask/255.0
  # mask = mask.astype(np.float32)
  mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
  train_masks.append(mask)

#Convert list to array for machine learning processing
train_masks = np.array(train_masks)

np.shape(train_images), np.shape(train_masks)

np.unique(train_masks)

import random
image_number = random.randint(0, 100)

image = train_images[image_number]
mask = train_masks[image_number]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image, cmap = 'jet')
axes[0].set_title('Image')
axes[0].set_axis_off()

axes[1].imshow(mask, cmap = 'gray')
axes[1].set_title('Mask')
axes[1].set_axis_off()

plt.show()

#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

np.shape(train_images), np.shape(train_masks_input)


image_number = random.randint(0, 100)

image = train_images[image_number]
mask = train_masks_input[image_number]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image, cmap = 'gray')
axes[0].set_title('Image')
axes[0].set_axis_off()

axes[1].imshow(mask, cmap = 'gray')
axes[1].set_title('Mask')
axes[1].set_axis_off()

plt.show()

from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.20, random_state = 0)

len(X1), len(y1), len(X_test), len(y_test)

# Normalize images by dividing by 255 (assuming they are in range [0, 255])
X_train_normalized = X_train.astype('float32') / 255.0
X_do_not_use_normalized = X_do_not_use.astype('float32') / 255.0

# Optional: Normalize the labels if needed, assuming they are binary (0 or 1)
# No normalization is usually needed for labels if they are already in [0, 1] range
y_train_normalized = y_train.astype('float32')
y_do_not_use_normalized = y_do_not_use.astype('float32')

# Check the lengths
print(len(X_train_normalized), len(y_train_normalized), len(X_do_not_use_normalized), len(y_do_not_use_normalized))

print("Class values in the dataset are ... ", np.unique(y_train), np.unique(y_train).__len__())  # 0 is the background/few unlabeled

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

n_classes = 83
activation = 'softmax'

LR = 0.0001
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)

# dice_loss = sm.losses.DiceLoss(class_weights = np.array([0.25, 0.25, 0.25, 0.25]))
# focal_loss = sm.losses.CategoricalFocalLoss()
# total_loss = dice_loss + (1 * focal_loss)



# Assuming y_true and y_pred are one-hot encoded categorical labels
# and num_classes is the number of classes in your problem (e.g., 4)
num_classes = 8

def dice_coefficient(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
    return tf.reduce_mean((2.0 * intersection + 1e-5) / (union + 1e-5))

def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    focal_loss = - (alpha * y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred) +
                    (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred))

    return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

def custom_loss(y_true, y_pred):
    dice_loss = 1 - dice_coefficient(y_true, y_pred)
    focal_loss = categorical_focal_loss(y_true, y_pred)

    total_loss = dice_loss + (1 * focal_loss)

    return total_loss

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

def model1(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='linear')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Create the model
model = model1()

# Summarize the model
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss=MeanSquaredError(),
              )


# Define the training parameters
batch_size = 64
epochs = 5

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
)

# Save the final model
model.save('unet_model.keras')  # Save the model in the new Keras format

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Load the trained model
model1()
# Function to evaluate the model on the test data
def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    loss = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")

    return loss

# Function to visualize predictions
def visualize_predictions(model, X_test, y_test, num_samples=5):
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(np.float32)  # Binarize predictions

    plt.figure(figsize=(15, num_samples * 5))

    for i in range(num_samples):
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(X_test[i], cmap='gray')
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(y_test[i].squeeze(), cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(predictions[i].squeeze(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


evaluate_model(model, X_test, y_test)

    # Visualize predictions
visualize_predictions(model, X_test, y_test, num_samples=5)
