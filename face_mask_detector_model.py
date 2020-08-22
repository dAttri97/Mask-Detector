import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
from imutils import paths
import os

data = []
labels = []

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

data = np.array(data,dtype='float32')
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX,testX, trainY,testY) = train_test_split(data,labels,test_size=0.2,stratify=labels,random_state=97)

aug = ImageDataGenerator(
    rotation_range = 20,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = 'nearest')

baseModel = MobileNetV2(weights='imagenet',include_top=False,input_tensor=Input(shape=(24,24,3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(128,activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2,activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layers in baseModel.layers:
    layers.trainable = False

#epoch = 20
#init_lr = 1e-4
#bs = 32
print('Compiling the Model...')
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4,decay = 1e-4/20),
              metrics = ['accuracy'])

print('Training Model...')

H = model.fit(
    aug.flow(trainX,trainY,batch_size=32),
    steps_per_epoch = len(trainX),
    validation_data = (testX,testY),
    validation_steps=len(testX),
    epochs = 20)

predict_indx = model.predict(testX,batch_size=32)
print('Predicting Data...')
predict_indx = np.argmax(predict_indx,axis=1)

print(classification_report(testY.argmax(axis=1),predict_indx,target_names=lb.classes_))

print('Saving the Model')
model.save('model',save_format='h5')


