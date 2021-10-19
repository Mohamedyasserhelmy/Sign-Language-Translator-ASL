# Sign Language Translator

A Deep Learning Application made with python to detect and recognize Sign Language gestures using "ASL" Standard then translates it into words.

## Description

We made this Application as our graduation project also we hope it could be used as an extension for video conference applications so that deaf and dump people can interact online easily with others.

The App Consists of 3 main modules.

1- Deep Learning Model.

2- Hand Detection.

3- Auto Correction. 

## Getting Started 

### Architecture

![Alt text](Assets/SimpleArch.png?raw=true "Title")

### Dataset Snapshots

![Alt text](Assets/DatasetSnapShots.jpg?raw=true "Title")

### Dataset Specifications

![Alt text](Assets/DatasetSpecs.jpg?raw=true "Title")

#### You can Download The Dataset using this link : `https://www.kaggle.com/grassknoted/asl-alphabet`

### Dependencies 

Note: you can run the command `pip install -r requirments.txt` and all libraries will be downloaded directly

Used Libraries: 

- MediaPipe 
- Numpy
- Tensorflow
- OpenCv-python
- pyspellchecker
- sklearn
- matplotlib

### Requirements

 - Camera (not less than 3 mega pixels)
 - CPU    (2.7 Ghz Multi-core Processor)
 - Python version >= 3.8  

### AI Model Specification

#### Models Accuarcy 

![Alt text](Assets/models_acc.png?raw=true "Title")

#### Model Architecture

![Alt Text](Assets/Model_arch.png)

## How to Run

 - You Can run the Application by simply openning `hand_detector` module then press f5.
 - You can also open `hand_detector` using python IDLE then from run window choose `run module`

## Demo

![Alt Text](Assets/demo.gif)

## Important Notes

- Model Trainning Process is Done seperately using Google Colab.
- To get the best efficiency from this application please make sure to run it in a bright place with plain background.
