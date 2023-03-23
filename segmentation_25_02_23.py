import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import segmentation_models as sm
from PIL import Image
from patchify import patchify
from tensorflow import keras
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler

DIR_PATH = "/content/drive/MyDrive/project/US_IMAGE_segmentation/Segmentation/Training Data/images" #path to original images


scaler = MinMaxScaler()
patch_size = 256

image_data = []

images = os.listdir(DIR_PATH)
for i, image_name in enumerate(images):
    #if image_name.endswith('.jpg'):
    image = cv2.imread(DIR_PATH+"/"+image_name, 1)
    width = (image.shape[1]//patch_size)*patch_size     #Nearest size divisible by our patch size
    height = (image.shape[0]//patch_size)*patch_size

    image = Image.fromarray(image)      # convert the input image into PIL image object
    image = image.crop((0, 0, width, height))   # crop from top left corner
    image = np.array(image)

    # Extract patches from each image

    #print('Now patchifying image:', path+"/"+image_name)
    patches = patchify(image, (patch_size, patch_size, 3), step=patch_size)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):

            single_patch_image = patches[i,j,:,:]

            single_patch_image = scaler.fit_transform(single_patch_image.reshape(-1, single_patch_image.shape[-1])).reshape(single_patch_image.shape)

            single_patch_image = single_patch_image[0]      #Drop the extra unecessary dimension that patchify adds
            image_data.append(single_patch_image)

MASK_DIR_PATH = "/content/drive/MyDrive/project/US_IMAGE_segmentation/Segmentation/Training Data/labels"
mask_data = []


masks = os.listdir(MASK_DIR_PATH)
for i, mask_name in enumerate(masks):
    #if image_name.endswith('.jpg'):
    mask = cv2.imread(MASK_DIR_PATH+"/"+mask_name, 1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    width = (mask.shape[1]//patch_size)*patch_size
    height = (mask.shape[0]//patch_size)*patch_size

    mask = Image.fromarray(mask)      # convert the input image into PIL image object
    mask = mask.crop((0, 0, width, height))
    mask = np.array(mask)

    # Extract patches from each mask
    
    #print('Now patchifying mask:', path+"/"+mask_name)
    patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)     # step=256 for 256 pixels, no overlap

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):

            single_patch_mask = patches_mask[i,j,:,:]
            
            # no need to scale a mask, can do it if want.
            #single_patch_mask = scaler.fit_transform(single_patch_mask.reshape(-1, single_patch_mask.shape[-1])).reshape(single_patch_mask.shape)
        
            single_patch_mask = single_patch_mask[0]    # Drop extra unecessary dimension that patchify adds. 
            mask_data.append(single_patch_mask)

len(image_data), len(mask_data)

 ###################################################

 """ Data Agumentation """

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os
from scipy.ndimage import rotate

import albumentations as A
images_to_generate=2000




"""images_path="/content/drive/MyDrive/project/US_IMAGE_segmentation/Segmentation/Training Data/images" #path to original images
masks_path = "/content/drive/MyDrive/project/US_IMAGE_segmentation/Segmentation/Training Data/labels" """
img_augmented_path="/content/drive/MyDrive/project/US_IMAGE_segmentation/Segmentation/agumentation/images" # path to store aumented images
mask_augmented_path="/content/drive/MyDrive/project/US_IMAGE_segmentation/Segmentation/agumentation/masks" # path to store aumented images
#images=[] 
#masks=[]

"""for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     
    images.append(os.path.join(images_path,im))

for msk in os.listdir(masks_path):  # read image name from folder and append its path into "images" array     
    masks.append(os.path.join(masks_path,msk))"""


aug = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=1),
    A.GridDistortion(p=1)
    ]
)

#random.seed(42)

i=1   # variable to iterate till images_to_generate


while i<=images_to_generate: 
    number = random.randint(0, len(image_data)-1)  #PIck a number to select an image & mask
    image = image_data[number]
    mask = mask_data[number]
    #print(image, mask)
    #image=random.choice(images) #Randomly select an image name
    #original_image = io.imread(image)
    #original_mask = io.imread(mask)
    
    augmented = aug(image=image, mask=mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']

    #images.append(transformed_image)
    #masks.append(transformed_mask)

        
    new_image_path= "%s/augmented_image_%s.png" %(img_augmented_path, i)
    new_mask_path = "%s/augmented_mask_%s.png" %(mask_augmented_path, i)
    io.imsave(new_image_path, transformed_image)
    io.imsave(new_mask_path, transformed_mask)
    i =i+1

######################################################

""" Loading Data """

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import keras
import segmentation_models as sm

img_augmented_path="/content/drive/MyDrive/project/US_IMAGE_segmentation/Segmentation/agumentation/images"
mask_augmented_path="/content/drive/MyDrive/project/US_IMAGE_segmentation/Segmentation/agumentation/masks"

image_data = []
mask_data = []

for img in os.listdir(img_augmented_path):
  image = cv2.imread(os.path.join(img_augmented_path, img), 1)    # reading image as 3 channel, because most of the pre-trained model looking image as 3 channel
  image = np.array(image)
  image_data.append(image)


for img in os.listdir(mask_augmented_path):
  mask = cv2.imread(os.path.join(mask_augmented_path, img), 0)
  mask = np.array(mask)
  mask_data.append(mask)

len(image_data), len(mask_data)

image_data = np.array(image_data)
mask_data = np.array(mask_data)

print(image_data.shape)
print(mask_data.shape)

# Check whether image and mask matches or not
import random
sample = random.randint(0, len(image_data))
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image_data[sample])
plt.subplot(122)
plt.imshow(mask_data[sample])
plt.show()

labels = np.expand_dims(mask_data, axis=3)
labels.shape

np.unique(labels)

# Separate Channel for each class in mask

from keras.utils import to_categorical

num_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=num_classes)

labels_cat.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_data, labels_cat, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

##############################################################3

"""# Unet From scratch"""

from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate
#from keras.layers.convolutional import Conv2D
from keras.models import Model

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)     # Not in original unet network
    x = Activation('relu')(x)

    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)     # Not in original unet network
    x = Activation('relu')(x)

    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2,2))(x)

    return x, p

def decoder_block(input, skip_feature, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_feature])
    x = conv_block(x, num_filters)

    return x


def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)       #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation ="softmax")(d4)

    model = Model(inputs, outputs, name="u_net")

    return model


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3

input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
num_classes = 6

model = build_unet(input_shape=input_shape, num_classes=num_classes)
model.compile(optimizer = 'adam', loss="categorical_crossentropy", metrics = ["accuracy"])
model.summary()

# Callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint("semantic_segmentation_aerial_satelite_images.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, mode='min')

history = model.fit(x_train, y_train, batch_size=1, epochs=25, validation_data=(x_test, y_test),
                    callbacks=[checkpoint_cb, early_stopping_cb], shuffle=False)


#####################################################


"""# Using Pre-Trained model"""


n_classes=6
activation='softmax'

LR = 0.0001
optim = tensorflow.keras.optimizers.Adam(LR)

metrics = [sm.metrics.IOUScore, sm.metrics.FScore(threshold=0.5)]

##################################################3

### Model 1

BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train1 = preprocess_input1(X_train)
X_test1 = preprocess_input1(X_test)

# define model
model1 = sm.Unet(BACKBONE1, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
#model1.compile(optim, total_loss, metrics=metrics)

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model1.summary())

# Callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint("'res34_backbone_cb.hdf5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, mode='min')

history1=model1.fit(X_train1, 
          y_train,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_test1, y_test),
          callbacks=[checkpoint_cb, early_stopping_cb])


model1.save('res34_backbone_50epochs.hdf5')

##############################################################

###Model 2

BACKBONE2 = 'inceptionv3'
preprocess_input2 = sm.get_preprocessing(BACKBONE2)

# preprocess input
X_train2 = preprocess_input2(X_train)
X_test2 = preprocess_input2(X_test)

# define model
model2 = sm.Unet(BACKBONE2, encoder_weights='imagenet', classes=n_classes, activation=activation)


# compile keras model with defined optimozer, loss and metrics
#model2.compile(optim, total_loss, metrics)
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model2.summary())

# Callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint("'inceptionv3_backbone_50epochs_cb.hdf5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')

history2=model2.fit(X_train2, 
          y_train,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_test2, y_test),
          callbacks=[checkpoint_cb, early_stopping_cb])


model2.save('inceptionv3_backbone_50epochs.hdf5')

##############################################################

###Model 3

BACKBONE3 = 'vgg16'
preprocess_input3 = sm.get_preprocessing(BACKBONE3)

# preprocess input
X_train3 = preprocess_input3(X_train)
X_test3 = preprocess_input3(X_test)


# define model
model3 = sm.Unet(BACKBONE3, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
#model3.compile(optim, total_loss, metrics)
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


print(model3.summary())

# Callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint("'vgg19_backbone_50epochs_cb.hdf5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')

history3=model3.fit(X_train3, 
          y_train,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_test3, y_test),
          callbacks=[checkpoint_cb, early_stopping_cb])


model3.save('vgg19_backbone_50epochs.hdf5')

###################################################################

"""# Ensemble models"""

from keras.models import load_model

#Set compile=False as we are not loading it for training, only for prediction.
model1 = load_model('saved_models/res34_backbone_50epochs.hdf5', compile=False)
model2 = load_model('saved_models/inceptionv3_backbone_50epochs.hdf5', compile=False)
model3 = load_model('saved_models/vgg19_backbone_50epochs.hdf5', compile=False)

#Weighted average ensemble
models = [model1, model2, model3]
#preds = [model.predict(X_test) for model in models]

pred1 = model1.predict(X_test1)
pred2 = model2.predict(X_test2)
pred3 = model3.predict(X_test3)

preds=np.array([pred1, pred2, pred3])

#preds=np.array(preds)
weights = [0.3, 0.5, 0.2]

#Use tensordot to sum the products of all elements over specified axes.
weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
weighted_ensemble_prediction = np.argmax(weighted_preds, axis=3)

y_pred1_argmax=np.argmax(pred1, axis=3)
y_pred2_argmax=np.argmax(pred2, axis=3)
y_pred3_argmax=np.argmax(pred3, axis=3)


#Using built in keras function
n_classes = 6
IOU1 = MeanIoU(num_classes=n_classes)  
IOU2 = MeanIoU(num_classes=n_classes)  
IOU3 = MeanIoU(num_classes=n_classes)  
IOU_weighted = MeanIoU(num_classes=n_classes)  

IOU1.update_state(y_test[:,:,:,0], y_pred1_argmax)
IOU2.update_state(y_test[:,:,:,0], y_pred2_argmax)
IOU3.update_state(y_test[:,:,:,0], y_pred3_argmax)
IOU_weighted.update_state(y_test[:,:,:,0], weighted_ensemble_prediction)


print('IOU Score for model1 = ', IOU1.result().numpy())
print('IOU Score for model2 = ', IOU2.result().numpy())
print('IOU Score for model3 = ', IOU3.result().numpy())
print('IOU Score for weighted average ensemble = ', IOU_weighted.result().numpy())

#Grid search for the best combination of w1, w2, w3 that gives maximum acuracy

import pandas as pd
df = pd.DataFrame([])

for w1 in range(0, 4):
    for w2 in range(0,4):
        for w3 in range(0,4):
            wts = [w1/10.,w2/10.,w3/10.]
            
            IOU_wted = MeanIoU(num_classes=n_classes) 
            wted_preds = np.tensordot(preds, wts, axes=((0),(0)))
            wted_ensemble_pred = np.argmax(wted_preds, axis=3)
            IOU_wted.update_state(y_test[:,:,:,0], wted_ensemble_pred)
            print("Now predciting for weights :", w1/10., w2/10., w3/10., " : IOU = ", IOU_wted.result().numpy())
            df = df.append(pd.DataFrame({'wt1':wts[0],'wt2':wts[1], 
                                         'wt3':wts[2], 'IOU': IOU_wted.result().numpy()}, index=[0]), ignore_index=True)
            
max_iou_row = df.iloc[df['IOU'].idxmax()]
print("Max IOU of ", max_iou_row[3], " obained with w1=", max_iou_row[0],
      " w2=", max_iou_row[1], " and w3=", max_iou_row[2])         



opt_weights = [max_iou_row[0], max_iou_row[1], max_iou_row[2]]

#Use tensordot to sum the products of all elements over specified axes.
opt_weighted_preds = np.tensordot(preds, opt_weights, axes=((0),(0)))
opt_weighted_ensemble_prediction = np.argmax(opt_weighted_preds, axis=3)

####################################################

#Predict on a few images

import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,:]
test_img_input=np.expand_dims(test_img_norm, 0)

#Weighted average ensemble
models = [model1, model2, model3]

test_img_input1 = preprocess_input1(test_img_input)
test_img_input2 = preprocess_input2(test_img_input)
test_img_input3 = preprocess_input3(test_img_input)

test_pred1 = model1.predict(test_img_input1)
test_pred2 = model2.predict(test_img_input2)
test_pred3 = model3.predict(test_img_input3)

test_preds=np.array([test_pred1, test_pred2, test_pred3])

#Use tensordot to sum the products of all elements over specified axes.
weighted_test_preds = np.tensordot(test_preds, opt_weights, axes=((0),(0)))
weighted_ensemble_test_prediction = np.argmax(weighted_test_preds, axis=3)[0,:,:]