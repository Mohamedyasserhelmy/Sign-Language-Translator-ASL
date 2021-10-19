from cnn import Model, DataGatherer
from tensorflow.keras.models import Sequential
from math import ceil
import matplotlib.pyplot as plt

training_dir = 'F:\\collegue\\GP project\\dataset\\asl_alphabet_train\\asl_alphabet_train'

#loading the images from training directory
data_gatherer = DataGatherer(training_dir)

x_train, x_test, y_train, y_test = data_gatherer.load_images()


batch_size = 64
training_size = x_train.shape[0]
test_size = x_test.shape[0]


#computing steps and validation steps per epoch according to training
#and testing size
compute_steps_per_epoch = lambda x: int(ceil(1. * x/batch_size))
steps_per_epoch = compute_steps_per_epoch(training_size)
val_steps = compute_steps_per_epoch(test_size)


#build the model
classifier = Model(Sequential()).classifier
classifier = Model.build_model(classifier)

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
history = classifier.fit(
  x_train, y_train,
  steps_per_epoch=steps_per_epoch,
  epochs=5,
  validation_data=(x_test, y_test),
  validation_steps=val_steps)

#plot accuracy graph
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='train_accuracy',)
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.title("classifier")

plt.show();

#run the below line to save the classifier
#Model.save(path, classifier)
