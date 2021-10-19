from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2


class Model:

  classifier = None
  def __init__(self, Type):
    self.classifier = Type
    
  
  def build_model(classifier):
    

    classifier.add(Convolution2D(128, (3, 3), input_shape=(64, 64, 1), activation='relu'))

    classifier.add(Convolution2D(256, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(256, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    classifier.add(Convolution2D(512, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5))

    classifier.add(Convolution2D(512, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5))

    classifier.add(Flatten())

    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(1024, activation='relu'))
    

    classifier.add(Dense(29, activation='softmax'))

    return classifier

  def save_classifier(path, classifier):
    classifier.save(path)

  def load_classifier(path):
    classifier = load_model(path)
    return classifier

  def predict(classes, classifier, img):
    img = cv2.resize(img, (64, 64))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0

    pred = classifier.predict(img)
    return classes[np.argmax(pred)], pred
    

class DataGatherer:

  def __init__(self, *args):
    if len(args) > 0:
      self.dir = args[0]
    elif len(args) == 0:
      self.dir = ""


  #this function loads the images along with their labels and apply
  #pre-processing function on the images and finaly split them into train and
  #test dataset
  def load_images(self):
    images = []
    labels = []
    index = -1
    folders = sorted(os.listdir(self.dir))
    
    for folder in folders:
      index += 1
      
      print("Loading images from folder ", folder ," has started.")
      for image in os.listdir(self.dir + '/' + folder):

        img = cv2.imread(self.dir + '/' + folder + '/' + image, 0)
        
        img = self.edge_detection(img)
        img = cv2.resize(img, (64, 64))
        img = img_to_array(img)

        images.append(img)
        labels.append(index)

    images = np.array(images)
    images = images.astype('float32')/255.0
    labels = to_categorical(labels)


    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)

    return x_train, x_test, y_train, y_test

  def edge_detection(self, image):
    minValue = 70
    blur = cv2.GaussianBlur(image,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res

