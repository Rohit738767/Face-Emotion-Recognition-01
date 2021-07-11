# Face-Emotion-Recognition-01
### Live Class Monitoring System(Face Emotion Recognition)
## Introduction
Emotion recognition is the process of identifying human emotion. People vary widely in their accuracy at recognizing the emotions of others. Use of technology to help people with emotion recognition is a relatively nascent research area. Generally, the technology works best if it uses multiple modalities in context. To date, the most work has been conducted on automating the recognition of facial expressions from video, spoken expressions from audio, written expressions from text, and physiology as measured by wearables.

Facial expressions are a form of nonverbal communication. Various studies have been done for the classification of these facial expressions. There is strong evidence for the universal facial expressions of seven emotions which include: neutral happy, sadness, anger, disgust, fear, and surprise. So it is very important to detect these emotions on the face as it has wide applications in the field of Computer Vision and Artificial Intelligence. These fields are researching on the facial emotions to get the sentiments of the humans automatically.

Here is the presentation link:https://github.com/Rohit738767/Face-Emotion-Recognition-01/blob/main/Deep%20Learning%20Presentation-converted.pdf

## Problem Statement
The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms.

Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms.
One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge.
In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention.

Digital classrooms are conducted via video telephony software program (ex-Zoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood. Because of this drawback, students are not focusing on content due to lack of surveillance.

While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analyzed using deep learning algorithms.

Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analyzed and tracked.

I will solve the above-mentioned challenge by applying deep learning algorithms to live video data.
The solution to this problem is by recognizing facial emotions.

## Dataset Information
I have built a deep learning model which detects the real time emotions of students through a webcam so that teachers can understand if students are able to grasp the topic according to students' expressions or emotions and then deploy the model. The model is trained on the FER-2013 dataset .This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.
Here is the dataset link:- https://www.kaggle.com/msambare/fer2013

## Dependencies
# Python 3
# Tensorflow
# Streamlit
# Streamlit-Webrtc
# OpenCV
# Deepface
# Model Creation
### 1) Using DeepFace
DeepFace is a deep learning facial recognition system created by a research group at Facebook. It identifies human faces in digital images. The program employs a nine-layer neural network with over 120 million connection weights and was trained on four million images uploaded by Facebook users.The Facebook Research team has stated that the DeepFace method reaches an accuracy of 97.35% ± 0.25% on Labeled Faces in the Wild (LFW) data set where human beings have 97.53%. This means that DeepFace is sometimes more successful than the human beings.

![download1](https://user-images.githubusercontent.com/74303124/125167105-8536e280-e1bc-11eb-929f-976647ee1f1e.png)


• The actual emotion in the Picture was ANGRY Face and then using DeepFace I found the prediction is ANGRY.

## 2) Using Deep Learning CNN
In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of deep neural network, most commonly applied to analyze visual imagery. They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on the shared-weight architecture of the convolution kernels or filters that slide along input features and provide translation equivariant responses known as feature maps.Counter-intuitively, most convolutional neural networks are only equivariant, as opposed to invariant, to translation. They have applications in image and video recognition, recommender systems, image classification, image segmentation, medical image analysis, natural language processing, brain-computer interfaces, and financial time series.CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. The "full connectivity" of these networks make them prone to overfitting data. Typical ways of regularization, or preventing overfitting, include: penalizing parameters during training (such as weight decay) or trimming connectivity (skipped connections, dropout, etc.) CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble patterns of increasing complexity using smaller and simpler patterns embossed in their filters. Therefore, on a scale of connectivity and complexity, CNNs are on the lower extreme. Convolutional networks were inspired by biological processes in that the connectivity pattern between neurons resembles the organization of the animal visual cortex. Individual cortical neurons respond to stimuli only in a restricted region of the visual field known as the receptive field. The receptive fields of different neurons partially overlap such that they cover the entire visual field.CNNs use relatively little pre-processing compared to other image classification algorithms. This means that the network learns to optimize the filters (or kernels) through automated learning, whereas in traditional algorithms these filters are hand-engineered. This independence from prior knowledge and human intervention in feature extraction is a major advantage.

![Cnn](https://user-images.githubusercontent.com/74303124/125167116-8ff17780-e1bc-11eb-9bd0-2e6dfe21edf7.jpg)


• The training gave the accuracy of 66% and test accuracy of 56%. It seems excellent. So, I save the model and the detection i got from live video was excellent.

• One drawback of the system is the some Disgust faces are showing Neutral .Because less no. of disgust faces are given to train .This may be the reason.

• I thought it was a good score should improve the score.

• Thus I decided that I will deploy the model.

## Loss & Accuracy Plot
![Loss accuracy plot](https://user-images.githubusercontent.com/74303124/125167127-997adf80-e1bc-11eb-8108-56bf1dbb776d.png)


To See the Training and Testing python file follow this link:https://github.com/Rohit738767/Face-Emotion-Recognition-01/blob/main/Face_recognition.ipynb

# Realtime Local Video Face Detection
I created one patterns for detecting and predicting single faces and as well as multiple faces using OpenCV videocapture in local.
For Webapp , OpenCV can’t be used. Thus, using Streamlit-Webrtc for front-end application.

## Deployment of Streamlit WebApp in Heroku and Streamlit
We deploy the app in streamlit but it invite the new user and it take time so after taht we use heroku and then we start deploy the app if you see the in the starting section you see the app name emotion45 but it not deploy we try as much as possible and then we put the link of app. But in local system it ran properly and app also fine

## Conclusion
Finally we build the webapp and deployed which has training accuracy of 66% and test accuracy of 56% . If you see how works visit link :-https://github.com/Rohit738767/Face-Emotion-Recognition-01/blob/main/Demo_local_video.mp4

## Some real life experience form project
Understand the deep concept of project

Don't afraid to faliure

From more faliure you get more experience and success will come

Never give up

Have some patience good things happen

Try new things and execute your idea
