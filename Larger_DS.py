#!/bin/python3

"""
Script to investigate training off of a larger sample
set. Will read in the first X images.

Uses new function annotations, so may only work in
Python > 3.9
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.constraints import unit_norm, max_norm, NonNeg
from sklearn.metrics import roc_curve
from astropy.io import fits
import astropy.visualization as avis
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random
import keract

plt.style.use("seaborn-darkgrid")

DATA_CSV = '/full/path/to/csv'
IMG_FOLD = '/full/path/to/images' #for me, ends with /Public/EUC_VIS/
DES_IMG_SIZE = 200
NUM_IMAGES = 50000 # can go up to nearly 100000, but there are a few entries with now images
BATCHSIZE = 200
EPOCHS = 20 # I liked doing 20 for quick ones, 40 for longer. maybe we should try more epochs?

# Creating the two normalization objects which
# can also work as functions later on in the
# data preprocessing
norm = avis.MinMaxInterval()
stretch = avis.AsinhStretch(0.010)


def read_csv(filename: str, num_images: int, to_skip: list[int] = []) -> pd.DataFrame:
    """
    Reads in the csv and trims it down to just the desired size and
    necessary columns. Also adds a column of booleans corresponding
    to whether a particular observation has been assigned to the
    training or testing pool. Training corresponding to True.
    """
    df = pd.read_csv(filename, skiprows=26)
    df = df[["ID", "n_sources", "n_source_im", "mag_eff", "n_pix_source"]]
    df["should_detect"] = (
        (df["n_source_im"] > 0) & (df["mag_eff"] > 1.6) & (df["n_pix_source"] > 20)
    )
    df = df[~df.ID.isin(to_skip)]  # clean bad rows w/o images
    df = df.sample(num_images) # Selects a random sample from the total df
    # Set boolean flags where True implies training set
    # Currently 4 in training for 1 in testing
    df["training"] = df.ID % 10 > 0
    train_df = df[df.training]
    test_df = df[~df.training]
    return train_df, test_df


def get_image(ID: int) -> np.array:
    """
    Reads in a specific image ID from the provided folder. Will
    crop to the desired size if smaller than the image size.
    """
    filename = f"{IMG_FOLD}imageEUC_VIS-{ID}.fits"
    img = fits.open(filename)
    center = img[0].data.shape[0] // 2
    image = img[0].data[
        center - DES_IMG_SIZE // 2 : center + DES_IMG_SIZE // 2,
        center - DES_IMG_SIZE // 2 : center + DES_IMG_SIZE // 2,
    ]
    return image


def package_data_training(train_df: pd.DataFrame) -> tuple[np.array, np.array]:
    """
    Generator function with reading in BATCHSIZE object from train_df at a time, extracts
    the necessary images, preprocesses them, and then yields the needed image and label
    array for fitting.
    """
    total_training_size = len(train_df)
    cursor = 0
    while True:
        batch_df = train_df.iloc[cursor : cursor + BATCHSIZE]
        actual_batch_size = len(batch_df)
        images_training = np.zeros((actual_batch_size, DES_IMG_SIZE, DES_IMG_SIZE))
        labels_training = batch_df["should_detect"].to_numpy()
        for i, ID in enumerate(batch_df.ID):
            image = get_image(ID)
            images_training[i] = pre_process_image(image)

        # Keras needs an extra dimension? So we add it here.
        images_training = np.expand_dims(images_training, axis=3)
        yield images_training, labels_training
        cursor = (cursor + BATCHSIZE) % total_training_size  # Cycling on to next batch


def package_data_testing(test_df: pd.DataFrame) -> tuple[np.array, np.array]:
    """
    Function to extract all the needed test images, preprocess them, and then return
    them along with their labels.

    TODO? Currently returns the entire batch of data all at once. For larger
    datasets, there might be enough images that this should be done with a
    generator as well.
    """
    testing_size = len(test_df)
    images_testing = np.zeros((testing_size, DES_IMG_SIZE, DES_IMG_SIZE))
    labels_testing = test_df["should_detect"].to_numpy()
    for i, ID in enumerate(test_df.ID):
        image = get_image(ID)
        images_testing[i] = pre_process_image(image)

    # Keras needs an extra dimension? So we add it here.
    images_testing = np.expand_dims(images_testing, axis=3)
    return images_testing, labels_testing


def pre_process_image(image: np.array) -> np.array:
    """
    Applies any preprocessing steps to stretch and normalize
    the image. Uses preconfigured normalization and stretch
    objects supplied be astropy.visualization.
    """
    return stretch(norm(image))


def build_keras_model(model_type: str = None) -> keras.Sequential:
    """
    Creates the desired CNN model. Passing in a known string will
    use preconfigured models taken from papers.
    """
    initializer = keras.initializers.HeNormal()
    model = keras.Sequential()
    if model_type == "davies":
        opts = {
            "strides": 2,
            "activation": "relu",
            "kernel_initializer": initializer,
            "padding": "same",
        }
        #lots of small conv w/ max pooling
        model.add(Conv2D(8, 15, **opts))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

        model.add(Conv2D(8, 15, **opts))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

        model.add(Conv2D(16, 5, **opts))
        model.add(Conv2D(16, 5, **opts))

        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=initializer))
        model.add(Dense(1, kernel_initializer=initializer, activation="sigmoid"))
    else:
        #my current model
        restraint = max_norm(3)
        convOpts = {
            "strides": 2,
            "activation": "relu",
            "kernel_initializer": initializer,
            "padding": "same",
            "kernel_constraint": restraint
        }
        maxOpts = {
            "strides": 2,
            "padding": "same",
        }
        model.add(Conv2D(64, 4, **convOpts))
        model.add(MaxPooling2D(4, **maxOpts))
        model.add(Conv2D(64, 4, **convOpts))
        model.add(MaxPooling2D(4, **maxOpts))

        model.add(Dropout(.5))
        model.add(Conv2D(64, 4, **convOpts))
        model.add(MaxPooling2D(4, **maxOpts))

        model.add(Dropout(.25))
        model.add(Conv2D(128, 2, **convOpts))
        
        model.add(Flatten())
        model.add(Dense(512,kernel_initializer=initializer,kernel_constraint=unit_norm()))
        model.add(BatchNormalization())
        model.add(Dense(1,kernel_initializer=initializer, activation="sigmoid"))

    model.compile(
        "adam",  # gradient optimizer
        loss="binary_crossentropy",  # loss function
        metrics=["accuracy"],  # what we are optimizing against
    )

    return model


def train_model(
    model: keras.Sequential,
    train_df: pd.DataFrame,
    validation: tuple[np.array, np.array],
    epochs: int = 10,
):
    """
    Fits the model using the given data.
    Currently setup for binary predictions, else
    the labels should have to_categorical applied
    to them.
    """
    history = model.fit(
        package_data_training(train_df),
        epochs=epochs,
        steps_per_epoch=int(np.ceil(len(train_df) / BATCHSIZE)),
        validation_data=(validation[0], validation[1]),
    )
    return history


def generate_roc(model: keras.Sequential, test_images: np.array, test_labels: np.array):
    """
    Creates a Receiver Operating Characteristic (ROC) curve for
    the success rate of the fitted model and plots it
    using matplotlib.
    """
    predicted = model.predict(test_images).ravel()
    false_pos_rate, true_pos_rate, thresh = roc_curve(test_labels, predicted)

    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(false_pos_rate, true_pos_rate)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"Fit using first {NUM_IMAGES} images over {EPOCHS} epochs.")
    plt.show()


def visualize_results(
    model: keras.Sequential,
    test_images: np.array,
    test_labels: np.array,
    test_df: pd.DataFrame,
    num_img: int = 9,
):
    """
    Visualizes a small sample of the testing images so that the normalized
    image can be viewed alongside its predicted value and with a border
    corresponding to whether it should have contained an event (green)
    or not (red).
    """
    sample_indices = random.sample(range(len(test_labels)), num_img)
    rows = int(np.ceil(num_img ** 0.5))
    cols = int(num_img ** 0.5)

    predicted = model.predict(test_images[sample_indices]).ravel()

    f = plt.figure(figsize=(15, 15))
    i = 0
    for i in range(num_img):
        ax = f.add_subplot(rows, cols, i + 1)
        ax.imshow(test_images[sample_indices[i]], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(test_df.iloc[i].ID)
        ax.grid(None)
        ax.set_axis_off()
        ax.text(
            DES_IMG_SIZE // 2,
            DES_IMG_SIZE - 5,
            f"{predicted[i]:0.3f}",
            color="white",
            horizontalalignment="center",
            fontsize=20,
        )
        if test_labels[sample_indices[i]]:
            col = "green"
        else:
            col = "red"
        rect = patches.Rectangle(
            (0, 0),
            DES_IMG_SIZE,
            DES_IMG_SIZE,
            lw=3,
            edgecolor=col,
            facecolor="none",
            clip_on=False,
        )
        ax.add_patch(rect)
    plt.tight_layout()
    plt.show()


def visualize_training_history(model_history):
    """
    Plots the training and testing accuracy over the course
    of all the training epochs. Useful for determining if a
    model is being over or underfit.
    """
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="test")
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def find_missing_images():
    """
    Simple function to systematically search through the given
    data csv and determine which entries are missing image data.

    Corresponding ID's written to skip_ids.txt.
    """
    df = pd.read_csv(DATA_CSV, skiprows=26)
    with open("data/skip_ids.txt", "w") as f:
        for _, row in df.iterrows():
            try:
                filename = f"{IMG_FOLD}imageEUC_VIS-{int(row.ID)}.fits"
                _ = fits.open(filename)
            except IOError:
                print("Missing image:", row.ID)
                f.write(f"{int(row.ID)}\n")


def read_in_rows_to_skip(filename: str) -> list[int]:
    """
    Reads in a file of image ID's to skip when generating a dataset. These
    ID's are missing a corresponding image, and thus can not be used.
    """
    with open(filename) as f:
        skip_ids = f.read().splitlines()
    return [int(i) for i in skip_ids]


def make_heatmap(model, images, labels):
    '''
    Use keract to visualize how the model interacts
    with an image. Picks a sample image
    where an event occurs and draws how each image works.
    TODO: this doesn't actually work here, or where it came
    from anymore. Why is it broken?
    '''
    label = 0
    while label == 0:
        num = random.randint(0,len(labels))
        label = labels[num]

    activation = keract.get_activations(model,images[num:num+1])
    #keract.display_activations(activation,fig_size=(10,10))
    keract.display_heatmaps(activation,images[num:num+1])

if __name__ == "__main__":
    to_skip = read_in_rows_to_skip("skip_ids.txt")
    train_df, test_df = read_csv(DATA_CSV, NUM_IMAGES, to_skip=to_skip)
    img_testing, labels_testing = package_data_testing(test_df)
    model = build_keras_model()
    history = train_model(
        model,
        train_df,
        (img_testing, labels_testing),
        epochs=EPOCHS,
    )
    os.system("say 'model training complete'") 
    model.summary()
    model.save("latest_model")
    visualize_training_history(history)
    generate_roc(model, img_testing, labels_testing)
    visualize_results(model, img_testing, labels_testing, test_df, 25)
    #make_heatmap(model,test_images,test_labels)

