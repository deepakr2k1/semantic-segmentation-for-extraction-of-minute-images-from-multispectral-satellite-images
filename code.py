
"""### Import required libraries"""

import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from zipfile import ZipFile
import albumentations as A
from datetime import datetime
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
from scipy.ndimage import rotate
import keras
from keras import backend, optimizers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras_unet_collection import models, losses
from tensorflow.keras.metrics import MeanIoU

# import zipfile
# with zipfile.ZipFile("/home/deepak/ship_masks","r") as zip_ref:
#     zip_ref.extractall("/home/deepak/dataset")
# with zipfile.ZipFile("/home/deepak/ship_images.zip","r") as zip_ref:
#     zip_ref.extractall("/home/deepak/dataset")

"""### Read Images"""

# Test images/masks path
X_path = "/home/deepak/dataset/images"
y_path = "/home/deepak/dataset/masks"

X = []
y = []

# read image & mask from folder and append it path into "original_image" and "original_mask" array respectively
for im in os.listdir(X_path):
    X.append(cv2.imread(os.path.join(X_path, im), 1))
    y.append(cv2.imread(os.path.join(y_path, im), 0))

print('Images Dataset size: ', len(X))
print('Masks Dataset size: ', len(y))

"""### Train Test Split"""

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)
print("Training dataset size:", len(X_train))
print("Training dataset size:", len(X_test))

"""### Preprocessing of data"""

# force channels-first ordering for all loaded images
backend.set_image_data_format('channels_last')

# Since our backbone model is a vgg16 , we convert the images from 750*750 to 256*256 dimension
SIZE = 256

# Training Images
images_train = []
masks_train = []

for i in range(len(X_train)):
    image = Image.fromarray(X_train[i])
    image = image.resize((SIZE, SIZE))    # resizing the image to 256*256
    images_train.append(np.array(image))   # appending the image converted to an array format to a list 

    image = Image.fromarray(y_train[i])
    image = image.resize((SIZE, SIZE))
    masks_train.append(np.array(image))

# Normalize images
X_train = np.array(images_train)/255.

# we aren't normalizing masks, just rescaling to 0 to 1.
y_train = np.expand_dims((np.array(masks_train)),3) /255.

# Testing Images
images_test = []
masks_test = []

for i in range(len(X_test)):
    image = Image.fromarray(X_test[i])
    image = image.resize((SIZE, SIZE))    # resizing the image to 256*256
    images_test.append(np.array(image))   # appending the image converted to an array format to a list 

    image = Image.fromarray(y_test[i])
    image = image.resize((SIZE, SIZE))
    masks_test.append(np.array(image))

# Normalize images
X_test = np.array(images_test)/255.

# we aren't normalizing masks, just rescaling to 0 to 1.
y_test = np.expand_dims((np.array(masks_test)),3) /255.

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[265], (256, 256, 3)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[265], (256, 256)), cmap='gray')
plt.show()

"""### Functions for Training Model & getting mean IoU"""

base_dir = "/home/deepak/Methods/M5/"
# os.mkdir(base_dir + "aug_X_trapped")
# os.mkdir(base_dir + "aug_y_trapped")

def train_Model(X_tr, y_tr, X_cv, y_cv):
    IMG_HEIGHT = X_tr.shape[1]
    IMG_WIDTH = X_tr.shape[2]
    IMG_CHANNELS = X_tr.shape[3]
    num_labels = 1
    batch_size = 8

    model = models.att_unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024],
                                 n_labels=num_labels,
                                 stack_num_down=2, stack_num_up=2,
                                 activation='ReLU',
                                 atten_activation='ReLU', attention='add',
                                 output_activation='Sigmoid',
                                 batch_norm=True, pool=False, unpool=False,
                                 backbone='DenseNet169', weights='imagenet',
                                 freeze_backbone=True, freeze_batch_norm=True,
                                 name='attunet')

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = 1e-2), 
                    metrics=['accuracy', losses.dice_coef])

    start_time = datetime.now()
    model_history = model.fit(X_tr, y_tr,
                                verbose=1,
                                batch_size = batch_size,
                                validation_data=(X_cv, y_cv ),
                                shuffle=False,
                                epochs=100)
    end_time = datetime.now()

      #Execution time of the model 
    execution_time_att_Unet = end_time - start_time
      #print("Attention UNet execution time: ", execution_time_att_Unet)

    return model

def train_Model(X_tr, y_tr, X_cv, y_cv):
    IMG_HEIGHT = X_tr.shape[1]
    IMG_WIDTH = X_tr.shape[2]
    IMG_CHANNELS = X_tr.shape[3]
    num_labels = 1
    batch_size = 8

    model = models.att_unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024],
                         n_labels=num_labels,
                         stack_num_down=2, stack_num_up=2,
                         activation='ReLU',
                         atten_activation='ReLU', attention='add',
                         output_activation='Sigmoid',
                         batch_norm=True, pool=False, unpool=False,
                         backbone='DenseNet169', weights='imagenet',
                         freeze_backbone=True, freeze_batch_norm=True,
                         name='attunet')

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = 1e-2), 
            metrics=['accuracy', losses.dice_coef])

    start_time = datetime.now()
    model_history = model.fit(X_tr, y_tr,
                        verbose=1,
                        batch_size = batch_size,
                        validation_data=(X_cv, y_cv ),
                        shuffle=False,
                        epochs=100)
    end_time = datetime.now()

    #Execution time of the model 
    execution_time_att_Unet = end_time - start_time
    print("Attention UNet execution time: ", execution_time_att_Unet)

    return model

# Return the Mean IoU
from tensorflow.keras.metrics import MeanIoU
import pandas as pd
def getMeanIoU(model, X_tst, y_tst):
  
    IoU_values = []
    for img in range(len(X_tst)):
        test_img = X_tst[img]
        ground_truth = y_tst[img]
        test_img_input = np.expand_dims(test_img, 0)
        prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

        IoU = MeanIoU(num_classes = 2)
        IoU.update_state(ground_truth[:,:,0], prediction)
        IoU = IoU.result().numpy()
        IoU_values.append(IoU)

    df = pd.DataFrame(IoU_values, columns=["IoU"])
    mean_iou = df.mean().values

    return mean_iou

"""### Training Model"""

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

model1 = train_Model(X_train, y_train, X_test, y_test)

model1.save(base_dir + "model1.hdf5")

model1 = tf.keras.models.load_model(base_dir + "model1.hdf5", compile=False)

# Calculate IoU and average
mean_iou = getMeanIoU(model1, X_train, y_train)
print("Mean IoU on training data: ", mean_iou)
mean_iou = getMeanIoU(model1, X_test, y_test)
print("Mean IoU on testing data: ", mean_iou)

"""### Capturing less IoU images and saving them"""

X_trapped_path = base_dir + "X_trapped"
y_trapped_path = base_dir + "y_trapped"

X_trapped = []
y_trapped = []

THRESHOLD = 0.65

less_iou_cnt = 0

for img in range(len(X_train)):
  from tensorflow.keras.metrics import MeanIoU
  
  test_img = X_train[img]
  ground_truth = y_train[img]
  test_img_input = np.expand_dims(test_img, 0)
  prediction = (model1.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
  
  IoU = MeanIoU(num_classes = 2)
  IoU.update_state(ground_truth[:,:,0], prediction)
  IoU = IoU.result().numpy()
  
  if IoU < THRESHOLD:
    X_trapped.append(X_train[img])
    y_trapped.append(y_train[img])
    
    # Saving the less IoU images
    img_name = "trapped_" + str(less_iou_cnt+1)+ ".png"
    (255 * X_trapped[0]).astype('int32') / 256
    cv2.imwrite(os.path.join(X_trapped_path, img_name), (255 * test_img).astype('int32') )
    cv2.imwrite(os.path.join(y_trapped_path, img_name), (255 * ground_truth))
    
    less_iou_cnt = less_iou_cnt + 1

print("Trapped Images: ", less_iou_cnt)

# Albumentation
import albumentations as A
import random
aug = A.Compose([
                 A.VerticalFlip(p=0.5),
                 A.HorizontalFlip(p=0.5),
                 A.RandomRotate90(p=0.5),
                 ])

aug_X_trapped_path = base_dir + "aug_X_trapped"
aug_y_trapped_path = base_dir + "aug_y_trapped"

aug_X_trapped = []
aug_y_trapped = []

images_generated = 0 # images_generated will keep the count of images generated
total_img_to_generate = 250 # Total number of images to generate

while images_generated < total_img_to_generate:

  # Pick a number to select an image & mask
  num = random.randint(0, len(X_train)-1)

  augmented = aug(image = X_train[num], mask = y_train[num])
  transformed_image = augmented['image']
  transformed_mask = augmented['mask']

  # append transformed images & masks
  aug_X_trapped.append(transformed_image)
  aug_y_trapped.append(transformed_mask)
    
  # Saving the augmented images & masks
  img_name = "aug_trapped_" + str(images_generated+1)+ ".png"
  cv2.imwrite(os.path.join(aug_X_trapped_path, img_name), (255 * transformed_image).astype('int32') )
  cv2.imwrite(os.path.join(aug_y_trapped_path, img_name), (255 * transformed_mask))

  # Update images_generated
  images_generated = images_generated + 1

    
# Append augment images & masks to X_train, y_train
X_train2 = X_train.copy()
y_train2 = y_train.copy()

for im in aug_X_trapped:
    np.append(X_train2, [im])
    
for im in aug_y_trapped:
    np.append(y_train2, [im])

"""### Retraining the Model after including augmented images"""

model2 = trainModel(X_train2, y_train2, X_test, y_test)

model2.save(base_dir + "model2.hdf5")

model2 = tf.keras.models.load_model(base_dir + "model2.hdf5", compile=False)

# Calculate IoU and average
mean_iou = getMeanIoU(model2, X_train, y_train)
print("Mean IoU on training data: ", mean_iou)
mean_iou = getMeanIoU(model2, X_test, y_test)
print("Mean IoU on testing data: ", mean_iou)

"""### Zooming the augmented trapped images"""

def zoomImg(img, mask):
  x1=len(mask)-1
  x2=0
  y1=len(mask)-1
  y2=0

  for i in range(len(mask)):
    for j in range(len(mask[i])):
      if((mask[i][j]).astype('int32')[0] == 1):
        x1 = min(x1, j)
        y1 = min(y1, i)
        x2 = max(x2, j)
        y2 = max(y2, i)

  if(x1 > x2):
      x1=0
      x2=len(mask)-1
      y1=0
      y2=len(mask)-1
  else:
      x1 = max(0, x1-32)
      x2 = min(len(mask)-1, x2+32)
      y1 = max(0, y1-32)
      y2 = min(len(mask)-1, y2+32)

  zoomed_img = []
  for i in range(y1, y2):
    temp = []
    for j in range(x1, x2):
      temp.append(img[i][j])
    zoomed_img.append(temp)

  zoomed_mask = []
  for i in range(y1, y2):
    temp = []
    for j in range(x1, x2):
      temp.append(mask[i][j])
    zoomed_mask.append(temp)
  
  return [zoomed_img, zoomed_mask]

res = zoomImg(X_train[265], y_train[265])

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(res[0], np.array(res[0]).shape), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(res[1], np.array(res[1]).shape), cmap='gray')
plt.show()

X_train3 = X_train.copy().tolist()
y_train3 = y_train.copy().tolist()

for i in range(len(aug_X_trapped)):
    res = zoomImg(aug_X_trapped[i], aug_y_trapped[i])

    image = Image.fromarray(np.array(res[0]).astype(np.uint8)).resize((SIZE, SIZE))
    X_train3.append(np.array(image))
    
    mask_shape = np.array(res[1]).shape
    mask = Image.fromarray(np.array(res[1]).reshape((mask_shape[0], mask_shape[1])).astype(np.uint8)).resize((SIZE, SIZE))
    mask = np.array(mask).reshape((SIZE, SIZE, 1))
    y_train3.append(mask)
    
X_train3 = np.array(X_train3)
y_train3 = np.array(y_train3)

print(X_train3.shape)
print(y_train3.shape)

"""### Retraining the Model after including zoomed images"""

model3 = trainModel(X_train3, y_train3, X_test, y_test)

model3.save(base_dir + "model3.hdf5")

model3 = tf.keras.models.load_model(base_dir + "model3.hdf5", compile=False)

# Calculate IoU and average
mean_iou = getMeanIoU(model3, X_train, y_train)
print("Mean IoU on training data: ", mean_iou)
mean_iou = getMeanIoU(model3, X_test, y_test)
print("Mean IoU on testing data: ", mean_iou)
