import os, random
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import tensorflow as tf

base_dir = '/home/ogai/Desktop/CNN-dataset' 
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'soiling')
train_dogs_dir = os.path.join(train_dir, 'clean')
batch_size = 10
train_size, validation_size, test_size = 850, 510, 50

img_width, img_height = 224, 224  # Default input size for model


def show_pictures(path):
  random_img = random.choice(os.listdir(path))
  img_path = os.path.join(path, random_img)
  img = image.load_img(img_path, target_size=(img_width, img_height))
  img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
  img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application
  plt.imshow(img_tensor)
  plt.show()

# Instantiate convolutional base
from keras.applications import MobileNetV2

conv_base = MobileNetV2(weights=None,#imagenet
                  include_top=False,
                  input_shape=(img_width, img_height, 3))  # 3 = number of channels in RGB pictures
# Freeze all
'''
for layer in conv_base.layers:
    layer.trainable = False
# Fine-tuning the model
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:  set_trainable = True
        if set_trainable:  layer.trainable = True
        else:  layer.trainable = False
'''
#conv_base.summary()

#pass our images through it for feature extraction
# Extract features
import os, shutil
from keras.preprocessing.image import ImageDataGenerator


#data generators
'''
train_datagen =  ImageDataGenerator(rescale=1./255,
                                    zoom_range=0.3,
                                    rotation_range=50,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,   horizontal_flip=True,
                                    fill_mode='nearest')


image_gen_train  = ImageDataGenerator (rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

datagen = ImageDataGenerator(rescale=1. / 255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
'''

image_gen_train  = ImageDataGenerator (rescale=1./255)

datagen = ImageDataGenerator(rescale=1./255)

def train_extract_features(directory, sample_count):
  # Must be equal to the output of the convolutional base
  features = np.zeros(shape=(sample_count, 7, 7, 1280))
  labels = np.zeros(shape=(sample_count))
  # Preprocess data
  generator = image_gen_train.flow_from_directory(directory,
                                          target_size=(img_width, img_height),
                                          shuffle=True,
                                          batch_size=batch_size,
                                          class_mode='binary')
  # Pass data through convolutional base
  def plotImages(images_arr):
      fig, axes = plt.subplots(1, 5, figsize=(20, 20))
      axes = axes.flatten()
      for img, ax in zip(images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
      plt.tight_layout()
      plt.show()

  augmented_images = [generator[0][0][0] for i in range(5)]
  plotImages(augmented_images)
  i = 0

  for inputs_batch, labels_batch in generator:
    features_batch = conv_base.predict(inputs_batch)
    features[i * batch_size:(i + 1) * batch_size] = features_batch
    # 上のコードでbatch_size = 20
    # だったとき
    # features[0:20], features[20:40], features[40:60], と増えていくのはわかるでしょうか。
    #
    # ここでは特徴（feature）と正解（label）のリストに
    # 指定フォルダから読み込んだデータをバッチサイズずつ詰め込んでいます。
    features[i * batch_size: (i + 1) * batch_size] = features_batch
    labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= sample_count:
      break
  return features, labels

def extract_features(directory, sample_count):
  features = np.zeros(shape=(sample_count, 7, 7, 1280))  # Must be equal to the output of the convolutional base
  labels = np.zeros(shape=(sample_count))
  # Preprocess data
  generator = datagen.flow_from_directory(directory,
                                          target_size=(img_width, img_height),
                                          batch_size=batch_size,
                                          class_mode='binary')
  # Pass data through convolutional base
  i = 0
  for inputs_batch, labels_batch in generator:
    features_batch = conv_base.predict(inputs_batch)
    # ここでは特徴（feature）と正解（label）のリストに
    # 指定フォルダから読み込んだデータをバッチサイズずつ詰め込んでいます。
    features[i * batch_size: (i + 1) * batch_size] = features_batch
    labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= sample_count:
      break
  return features, labels


train_features, train_labels = train_extract_features(train_dir, train_size)  # Agree with our small dataset size
validation_features, validation_labels = extract_features(validation_dir, validation_size)
test_features, test_labels = extract_features(test_dir, test_size)



#on top of our convolutional base,
#we will add a classifier and then our model is ready to make predictions.
from keras import models
from keras import layers
from keras import optimizers

EPOCHS = 300

model = models.Sequential()
model.add(layers.Flatten(input_shape=(7, 7, 1280  )))
model.add(layers.Dense(256, activation='relu', input_dim=(7, 7, 1280)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
# Compile model
base_learning_rate = 0.0001
model.compile(optimizer=optimizers.RMSprop(lr=base_learning_rate),
              # optimizers.Adam(),
              # optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

np.random.seed(100)
np.random.shuffle(train_features)
np.random.seed(100)
np.random.shuffle(train_labels)

np.random.seed(100)
np.random.shuffle(validation_features)
np.random.seed(100)
np.random.shuffle(validation_labels)

# Train model
#fit(x,y) x is input data, y is numpy array labels
history = model.fit(train_features, train_labels,
                    epochs=EPOCHS,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels),
                    verbose=1)


# Save model
model.save('/home/ogai/Desktop/CNN-dataset/test1_noGAN.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


test_cats_dir =  '/home/ogai/Desktop/CNN-dataset/test/soiling'
test_dogs_dir =  '/home/ogai/Desktop/CNN-dataset/test/clean'

# Define function to visualize predictions
class_names = np.array(test_labels)

def visualize_predictions(classifier, n_cases):
    plt.figure(figsize=(10, 9))
    for i in range(0,n_cases):
        path = random.choice([test_cats_dir, test_dogs_dir])
        # Get picture
        random_img = random.choice(os.listdir(path))
        img_path = os.path.join(path, random_img)
        img = image.load_img(img_path, target_size=(img_width, img_height))
        img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
        img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application
        # print("img_path:",img_path,"random_img:",random_img,"img_tensor",img_tensor)
        # Extract features
        features = conv_base.predict(img_tensor.reshape(1,img_width, img_height, 3))
        # Make prediction
        try:
            prediction = classifier.predict(features)
        except:
            prediction = classifier.predict(features.reshape(1, 7*7*1280))

        # Show picture
        plt.subplot(5, 20, i + 1)
        plt.subplots_adjust(hspace=0.3)
        plt.axis('off')
        plt.imshow(img_tensor)
        _ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")

        if '.png' in random_img:
            labels_img = 0
        else:
            labels_img = 1
        color = "blue"
        # Write prediction
        if prediction < 0.5:
            prediction_labels = 0
            color = "blue" if labels_img == prediction_labels else "red"
            plt.title("clean",color = color)
            print('clean')
        else:
            prediction_labels = 1
            color = "blue" if labels_img == prediction_labels else "red"
            plt.title("soiling",color = color)
            print('soiling')
        score = model.evaluate(test_features,test_labels,verbose=1)
        print('The loss:',score[0])
        print('The accuracy:',score[1])
    plt.show()




# Visualize predictions
visualize_predictions(model, 100)