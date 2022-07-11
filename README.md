# SETSCI Article English: This is the translation of our Turkish language published article to English language.



<h1 align="center">Development of a Deep Learning Based Model for Recognizing the Environmental Sounds in Videos</h1>

**Abstract** — Nowadays, decomposition of various environmental sounds for environment recognition has gained popularity.
Various background sounds in videos could be classified with high success with deep learning and machine learning techniques.
In this way, semantically enriched video scenes can be depicted. This work contains the process of developing a convenient deep
learning neural network model for environmental sounds recognition. In training the developed model, ten main categories have
been chosen from a dataset that has various data to test the model's prediction power by experiment. From the limited data
available, first, spectrograms have been produced and then, these spectrograms have been enriched by the help of data
augmentation techniques. In this way, attribute diversity that was gained from data has been increased. Also, with three different
design approaches for training the model, source codes have been written. These codes have been created by using deep learning
network model-based methods such as Convolutional Neural Networks, Long Short Term Memory, Gated Recurrent Unit. Seven
different designed neural network models have been trained by experiments and achievement has been proved by tests. With the
highest accuracy obtained from one of the generated models, approximately %87 of accuracy has been obtained. This work
contains obtained experimental results and scientific evaluation.

**Keywords** --- Convolutional Neural Networks, Recurrent Neural Networks, Environmental Sound Recognition

<p align="center"> <strong> I. Introduction </strong> </p>

  Sound Recognition, which emerged with the development of technology, is a technology based on the analysis of the audio signal. This process can be performed using Deep Learning methods like many other methods. To these methods; Methods such as creating Convolutional Neural Networks (CNN) by creating a spectrogram, and establishing Recurrent Neural Networks (RNN) using Time Series can be given as examples. Voice recognition processes can be used in different areas. Detection of sound scene events for multimedia systems [1], [2], [3], [4], conversion of the listened sound to text (Speech to Text) can be given as examples of these areas. However, the data set to be used for the voice recognition method is usually limited. Therefore, data augmentation methods are expected to be performed very efficiently before training the model. As the data in the dataset increases, the expected performance from the trained model will increase, so the efficiency of the model also increases. Video pieces are used in studies as a very rich environment in terms of environmental sounds, due to the visual and auditory elements they contain. Environmental Sound Recognition, on the other hand, is a wide field that covers all kinds of sounds that can be heard from the environment in daily life and includes various sound attributes [5], [6], [7], [8]. Examples of these sounds are many sounds such as baby crying, wave sound. 
  In this study, the literature review in the second part, the data set in the third part, and the models created in our study from the fourth to the twelfth part are included. By using different techniques to increase the data created with the available feature set by converting the environmental sounds into spectrograms in our models, different deep learning models were created and these models were trained with the available data. In the twelfth chapter, the experimental results are given, and in the thirteenth chapter, the results and discussions.
  
<p align="center"> <strong> II. Literature Review </strong> </p>

  With the development of the Deep Learning method and its effective use in areas such as Computer Vision, many studies have been carried out in the literature to find solutions to such problems. Architectural models created with CNN, which is a widely used architecture within the scope of deep learning, and by starting to get effective results in classification problems with these models, and CNN has been used frequently for the solution of problems such as voice recognition.
  The deep neural network is trained on piecewise spectrograms with the model proposed by Piczak [2] for solving the classification problem related to sound. The log-mel properties of each sound frame (inspected sound window) were obtained as sound features and these spectrograms were given to the CNN model. Using the ESC-10 dataset, an accuracy rate of 81.5% was obtained as a result of this training.
  Shaobo et al. with the RawNet model proposed in his study [3], the CNN model was trained using spectrograms created with raw waveform rather than log-mel feature as input. As a result of the training using the ESC-10 dataset, an accuracy rate of 85.2% was obtained.
  Khamparia et al. two different models were created using CNN and Tensor Deep Stacking Networks (TDSN) by obtaining log-mel feature spectrograms with the Khamparia model proposed in his study [4]. With the model created using the CNN infrastructure, an accuracy rate of 77% was obtained in the ESC-10 dataset. With the model created using the other model, TDSN, an accuracy rate of 56% was obtained in the ESC-10 dataset.
  Krishevsky et al. In the study [5], spectrograms with log-mel properties were obtained with the proposed AlexNet model. The feature variety was increased by The method of obtaining MFCC (Mel-frequency cepstral coefficients) [6] type features as the attribute type consisting of coefficients in the Mel scale and the Cross Recurrence Plot (CRP) [9] method, which includes cross repetitions. Then, the spectrograms obtained from these three methods were combined under a single spectrogram as a single color. With the data set obtained in this way, the model was created using Convolutional Recurrent Neural Networks (ETSA). This model reached an accuracy rate of 86% as a result of deep neural network training using the ESC-10 dataset.
  
<p align="center"> <strong> III. Dataset </strong> </p>















