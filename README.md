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

In order to create the structures and models designed in our study, ten sound categories were used from the dataset containing a total of fifty sound classes consisting of five main subject categories and including environmental sounds [10]. These categories are: Animals, Natural soundscapes & water sounds, Human, non-speech sounds, Interior/Domestic sounds, Exterior/Urban noises categories. Each of these categories has a total of 10 sound classes. Sounds in these categories are: “Chirping birds”, “Thunderstorm”, “Breathing”, “Brushing teeth”, “Can opening”, “Clock tick”, “Chainsaw”, “Car horn”, “Church bells”, “Airplane” . For each sound, there are forty different sound samples. The length of these audio recordings is about five seconds. Background sound was added to the available sounds to increase the dataset. This background sound is an audio track containing one minute of white noise. A 5000 ms segment was taken from this background sound to be added to the five-second sounds at hand. Later, this section was integrated into the sounds. In this way, the data size in the data set is doubled. The modified dataset obtained here is used as the basic dataset of the model. In addition, according to the model to be established, methods such as MFCC type feature acquisition method, Chroma [7] type feature acquisition method, Multiple Masking [11] are applied on this basic dataset to increase the variety of features to be obtained from the dataset. In general, the dataset in our study are divided 80% training, 10% validation, 10% testing.

<p align="center"> <strong> IV. Used Libraries and Environment for Developing the Model </strong> </p>

  Keras [12] library was used to create all deep learning models. In addition, pre-trained ResNet50 [13], MobileNetV2 [14], VGG16 and VGG19 [15] models, which are ready in the Keras library and used as the base model of the given model, were used. The creation of the model is based on the spectrogram of the available audio recording with the help of both the Librosa [8] and Tensorflow [16] libraries. These spectrograms were created by taking log-mel properties. Then, with the help of these libraries, data augmentation was provided programmatically for the available data set. The SoundNet [17], [18] library, which provides direct decomposition of sounds over the given input video, has also been tested. With the help of the Matplotlib [19] library, the generated spectrograms were turned into a graph. Also Numpy [19], Pandas [19], OS [19] etc. libraries were also used. During the trainings and tests, one of them is Intel Core i7-8750H processor with 16 GB RAM memory, running at 2.20 GHz, Nvidia GeForce GTX 1060 graphics card hardware features, and the other is Intel Core i7-9750H processor with 8 GB RAM memory, running at 2.6 GHz and Two different laptop computers with Nvidia GeForce GTX 1650 graphics card hardware features were used. Figure 1 shows some examples of spectrograms used in our study.
  
 BURAYA RESİM!!!!!!

<p align="center"> <strong> V. Used Libraries and Environment for Developing the Model </strong> </p>

The dataset was converted into a spectrogram using the method available in the Librosa [8] library to create a spectrogram. Then, the dataset for this model is exceptionally divided into 70% training, 20% validation and 10% testing. The design is as follows, on the layer structure of the pre-trained "MobileNetV2" [14] neural network model, a two-dimensional GlobalAveragePooling2D layer is added. Afterwards, a 512-unit Hidden Layer was created. Dropout has been performed and a ten-unit Output Layer has been created. In our study, this model seen in Table 1 was named "Design1".

_Table 1. The architectural structure of the first design created._

<table>
  <tr>
    <th> MobileNetV2 ve Convolutional Structure </th>
  </tr>
  <tr>
    <td> MobileNetV2 </td>
  </tr>
  <tr>
    <td> GlobalAveragePooling2D </td>
  </tr>
  <tr>
    <td> 512-unit Hidden Layer-ReLU </td>
  </tr>
  <tr>
    <td> Dropout 0.5 </td>
  </tr>
  <tr>
    <td> 10 Dense-Softmax </td>
  </tr>
</table>

<p align="center"> <strong> VI. Result Obtained with Using SoundNet </strong> </p>

SoundNet model and its library is a ready-made voice recognition model in the literature created using [17], [18], Environmental Sound Classification-50 dataset (ESC-50) [10] and Detection and Classification of Acoustic Scenes and Events dataset (DCASE)  [20]. In order to be given to this model, a three-minute section was taken from the gameplay video [21] of a computer game with environmental sounds and given to this model. However, the estimate obtained was not satisfactory. For example, in a part of the video: 43% probability of being in a television studio, 12% probability of being in an aquarium, 24% of skiing were estimated, but none of these events took place in that part. Only predictions could be made to describe accurate environmental scenes in certain parts. For this reason, as mentioned here, the SoundNet infrastructure was also examined and evaluated experimentally so that the model we developed could obtain more reliable estimation results.

<p align="center"> <strong> VII. Incresing the Dataset by Obtaining MFCC-Type Features </strong> </p>

Since the available dataset is limited, increasing the feature set will enable the model to give better experimental results. For this reason, in order to increase the feature set, the MFCC type sound features acquisition method was applied to the audio segments in the form of spectrogram images with the help of the Librosa [8] library and the feature set was increased. Later, the same design and the MobileNetV2 [14] neural network model were used. In our study, this structure was named "Design2".

<p align="center"> <strong> VIII. Increase the Attribute Set by Using Sound Attributes Together </strong> </p>

Chroma-type feature acquisition method with the help of Librosa [8] library was applied to the sounds in the dataset, which is another method to increase the feature types in the dataset, and it was added to the existing data previously created by the MFCC type feature acquisition method. The new model was trained with the increased data. In our study, this structure was named "Design3".

<p align="center"> <strong> IX. Obtained Model with Using Multiple Masking </strong> </p>

In order to increase the data in the dataset, spectrogram was taken with the help of Tensorflow [16] library, which is another method, and Multiple Masking [11] was applied. The Multiple Masking process [22] expands the training distribution by plotting the shear plane from random parts of a spectrogram according to the given parameters to increase and mix the data. Frequency mask parameter 36, time mask parameter 24 is taken. These values have been chosen because the most appropriate masks are created. For each given spectrogram, 2 frequency and 2 time masks were created. First of all, some of the sounds were cut and new data were obtained by mixing the sounds in the common category. For example, the data set has been increased with the new data created by combining the "Breathing" and "Tooth Brushing" sounds in the category of "Sounds Not Containing Human Speech". While creating this model, a study in the literature was used, in which the model created using this method was used before [23]. Multiple Masking process is applied to only the spectrograms obtained at the beginning were used to the dataset, MFCC [6] and Chroma [7] type feature acquisition method are not been used and the model was trained. In our study, this structure, examples of which can be seen in Figure 2, was named "Design4".

 BURAYA RESİM!!!!!!
 
<p align="center"> <strong> X. Data Given to the Programatically Created Model </strong> </p>

The programmatic implementation of the design model, which was developed based on the model [24] in a study in the literature, works as follows, thanks to its coding in Python language. First, a CNN model with four Convolutional Layers is created. Batch Normalization is done after each Convolutional Layer is created. A Pooling operation takes place in both Convolutional Layers. In the models in our study, the relevant activation processes are also performed using the Rectified Linear Unit (ReLU). After the last Pooling process, a model is created for two Gated Recurrent Units (KTB) with hyperbolic tangent (tanh) activation. After the Dropout process, the “Softmax” process is performed and the classification result is obtained. The dataset given to this programmatic structure is the same as the dataset mentioned in the previous section. In our study, this model seen in Table 2 was named "Design5".

_Table 2. CNN and RNN design architecture built using four Convolutional Layers._

<table>
  <tr>
    <th> Convolutional RNN </th>
  </tr>
  <tr>
    <td> 3x5x32 Convolutional1-Batch Normalization-ReLU </td>
  </tr>
  <tr>
    <td> 3x5x32 Convolutional2-Batch Normalization-ReLU </td>
  </tr>
  <tr>
    <td> 4x3 MaxPooling </td>
  </tr>
  <tr>
    <td> 3x1x64 Convolutional3-Batch Normalization-ReLU </td>
  </tr>
  <tr>
    <td> 3x1x64 Convolutional4-Batch Normalization-ReLU </td>
  </tr>
  <tr>
    <td> 4x1 MaxPooling </td>
  </tr>
  <tr>
    <td> 256 GRU1-tanh-Dropout 0.5 </td>
  </tr>
  <tr>
    <td> 256 GRU2-tanh-Dropout 0.5 </td>
  </tr>
  <tr>
    <td> 10 Fully Connected Layer - Softmax </td>
  </tr>
</table>

<p align="center"> <strong> XI. Increasing the Number of Convolutional Layers in the New Design </strong> </p>

In the programmatic design, the number of Convolutional Layers has been increased to eight by arranging the four-layer code. The model was trained with a fifty training epoch. It has been decided not to use GRU layers because the result is worse when GRU layers are added. In our study, this model seen in Table 3 was named "Design6".

_Table 3. CNN and RNN main design architecture (before removing the GRU layers) built using eight Convolutional Layers._

<table>
  <tr>
    <th> Convolutional RNN </th>
  </tr>
  <tr>
    <td> 3x5x32 Convolutional1-Batch Normalization-ReLU </td>
  </tr>
  <tr>
    <td> 3x5x32 Convolutional2-Batch Normalization-ReLU </td>
  </tr>
  <tr>
    <td> 4x3 MaxPooling </td>
  </tr>
  <tr>
    <td> 3x1x64 Convolutional3-Batch Normalization-ReLU </td>
  </tr>
  <tr>
    <td> 3x1x64 Convolutional4-Batch Normalization-ReLU </td>
  </tr>
  <tr>
    <td> 4x1 MaxPooling </td>
  </tr>
  <tr>
    <td> 1x5x128 Convolutional5-Batch Normalization-ReLU </td>
  </tr>
  <tr>
    <td> 1x5x128 Convolutional6-Batch Normalization-ReLU </td>
  </tr>
  <tr>
    <td> 1x3 MaxPooling </td>
  </tr>
  <tr>
    <td> 3x3x256 Convolutional7-Batch Normalization-ReLU </td>
  </tr>
  <tr>
    <td> 3x3x256 Convolutional8-Batch Normalization-ReLU </td>
  </tr>
  <tr>
    <td> 2x2 MaxPooling </td>
  </tr>
  <tr>
    <td> 256 GRU1-tanh-Dropout 0.5 </td>
  </tr>
  <tr>
    <td> 256 GRU2-tanh-Dropout 0.5 </td>
  </tr>
  <tr>
    <td> 10 Fully Connected Layer - Softmax </td>
  </tr>
</table>

<p align="center"> <strong> XII. Obtained Model With Using VGG19 and RNN </strong> </p>

It is given to RNN with (LSTM) layers. Then the attention process is applied. The Permute process and the 128-unit Dense creation part in the coding architecture of the design were created within the framework of this attention event. Finally, the result of the inputs to the Dense Layers created with the "Softmax" function comes out in the Fully Connected Layer. The model is trained with the data given to the implementation of the programmatic design created with this logic. In our study, this model seen in Table 4 was named "Design7".

_Table 4. Creating an RNN model on the model created with VGG19 and examining different points of the inputs to this model with the attention event._

<table>
  <tr>
    <th> VGG19 & RNN </th>
  </tr>
  <tr>
    <td> VGG19 </td>
  </tr>
  <tr>
    <td> 128x384 Input </td>
  </tr>
  <tr>
    <td> 256 Bi-LSTM </td>
  </tr>
  <tr>
    <td> 2x1 Permute </td>
  </tr>
  <tr>
    <td> 128-unit Dense ReLU </td>
  </tr>
  <tr>
    <td> Flatten </td>
  </tr>
  <tr>
    <td> 512-unit Dense ReLU </td>
  </tr>
  <tr>
    <td> Dropout 0.5 </td>
  </tr>
  <tr>
    <td> 10 Fully Connected Layer - Softmax </td>
  </tr>
  </table>

<p align="center"> <strong> XIII. Experimental Results </strong> </p>

When the results of the experiments are examined, the model with the highest test accuracy and the lowest loss value is the model created using eight Convolutional Layers and without using GRU. As can be seen from Table 5, an accuracy rate of 87% and a loss of 0.88 were obtained as a percentage in the test results with the best model, "Design6". This shows that the model has an acceptable success. Each model created is a change that has been created or made on the previous model to improve the results obtained in previous models.

_Table 5. Obtained accuracy and loss values from generated models._

<table>
  <tr>
    <th  colspan="3"> Obtained Results from Experiments </th>
  </tr>
  <tr>
    <th> Model </th>
    <th> Accuracy Rate </th>
    <th> Training Loss Value </th>
  </tr>
  <tr>
    <td> Design1 </td>
    <td> 0,60 </td>
    <td> 1,37 </td>
  </tr>
  <tr>
    <td> Design2 </td>
    <td> 0,74 </td>
    <td> 1,73 </td>
  </tr>
  <tr>
    <td> Design3 </td>
    <td> 0,59 </td>
    <td> 1,08 </td>
  </tr>
  <tr>
    <td> Design4 </td>
    <td> 0,71 </td>
    <td> 0,81 </td>
  </tr>
  <tr>
    <td> Design5 </td>
    <td> 0,73 </td>
    <td> 1,24 </td>
  </tr>
  <tr>
    <td> Design6 </td>
    <td> 0,87 </td>
    <td> 0,88 </td>
  </tr>
  <tr>
    <td> Design7 </td>
    <td> 0,42 </td>
    <td> 1,82 </td>
  </tr>
  </table>

Accuracy rate is one of the objective criteria commonly used to determine the class discrimination ability of the classifier on the dataset in an experiment. According to the confusion matrix table in the literature; true positive (TP), false positive
(FP), evaluated by true negative (TN) and false negative (FN) measurements, is given by Equation (1) below. This value can be shown both as a decimal and expanded as a percentage value in studies [25].

<p align="center"> Accuracy = {TP+TN}{TP+TN+FP+FN} (1) </p>

The loss value obtained in the experiments is useful in following the changes in the objective function of the relevant deep neural network model during the training epoch and in understanding the quality of the training at the end of the training. In Table 6, a comparison of the experimental results of the model named "Design6" in this study with the similar studies in this field in the literature is given.

_Table 6. Competing for success with studies in the literature._

<table>
  <tr>
    <th> Model </th>
    <th> Used Structure & Feature </th>
    <th> Accuracy Rate </th>
  </tr>
  <tr>
    <td> Khampdaria [4] </td>
    <td> log-mel, TDSN </td>
    <td> 56% </td>
  </tr>
  <tr>
    <td> Khampdaria [4] </td>
    <td> log-mel, CNN </td>
    <td> 77% </td>
  </tr>
  <tr>
    <td> Piczak [2] </td>
    <td> log-mel, CNN </td>
    <td> 81,5% </td>
  </tr>
  <tr>
    <td> RawNet [3] </td>
    <td> raw waveform, CNN </td>
    <td> 85,2% </td>
  </tr>
  <tr>
    <td> AlexNet [5] </td>
    <td> log-mel, MFCC, CRP, CRNN </td>
    <td> 86% </td>
  </tr>
  <tr>
    <td> Our Work (Design6) </td>
    <td> log-mel, Multiple Masking, CNN </td>
    <td> 87% </td>
  </tr>
  </table>

As can be seen from Table 6, it is seen that the best result among the CNN models created by only taking the log-mel properties of the spectrograms, as in the model in this study, is in the model in this study. Using TDSN instead of using CNN leads to much worse performance in training this dataset. Besides using MFCC attributes, using data augmentation methods such as CRP and replacing CNN to CRNN is also effective in increasing the accuracy of the model.

<p align="center"> <strong> XIV. Results AND Discussion </strong> </p>

In this study, by increasing the variety of features obtained from the data, the semantic gap between low-level features and high-level features has been reduced both in data enrichment and neural network training. For a realistic video experience in our study, considering the results obtained in the sound scene estimation of the video game [21] of the video game mentioned in the SoundNet [17, 18] experiment above, with the model we developed, in the scene in a part of this video: Similarly, when estimating that there is a car horn sound and a scene with a 1% probability of airplane noise, estimations were made that there was actually a car sound in this scene.

It is understood that the implementations of the new design models designed in our study can be used successfully in the classification and prediction of various types of sound scenes and events. In this respect, these new deep neural network models, designed and implemented by us, are suggested as the main contribution to the literature in our study. In our future work, it is aimed to find solutions to the problems of recognizing sound scenes and sound events on a wider scale by using these models.

<p align="center"> <strong> References </strong> </p>

-[1] B., Karasulu. “Çoklu Ortam Sistemleri İçin Siber Güvenlik
Kapsamında Derin Öğrenme Kullanarak Ses Sahne ve Olaylarının
Tespiti” ACTA INFOLOGICA, 3(2):60-82, 2019. doi:
10.26650/acin.590690

-[2] K. J. Piczak, “Environmental sound classification with convolutional
neural networks”, 2015 IEEE 25th International Workshop on
Machine Learning for Signal Processing (MLSP), Boston, MA, USA
pp. 1-6. 2015. doi: 10.1109/MLSP.2015.7324337

-[3] L., Shaobo, Y., Yao, J., Hu, G., Liu, X. Yao, and J., Hu. “An
ensemble stacked convolutional neural network model for
environmental event sound recognition”, Applied Sciences, vol. 8, no.
7 (2018): 1152. 2018. doi: 10.3390/app8071152

-[4] A., Khamparia, D., Gupta, N.G., Nguyen, A., Khanna, B., Pandey and
P., Tiwari, “Sound Classification Using Convolutional Neural
Network and Tensor Deep Stacking Network”, IEEE Access, vol. 7,
pp. 7717-7727, 2019. doi: 10.1109/ACCESS.2018.2888882

-[5] A. Krizhevsky, I. Sutskever, G.E. Hinton. “ImageNet Classification
with Deep Convolutional Neural Networks”, Advances in Neural
Information Processing Systems, Editörler: F. Pereira and C.J. Burges
and L. Bottou and K.Q. Weinberger, Curran Associates, Inc., Vol. 25,
2012. [Online]: https://proceedings.neurips.cc/paper/2012/file/
c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

-[6] MFCC (Mel-frequency cepstral coefficient) dokümantasyonu, 2022.
https://librosa.org/doc/main/generated/librosa.feature.mfcc.html

-[7] Chroma ses özniteliği dokümantasyonu, 2022. [Online].
https://librosa.org/doc/main/generated/librosa.feature.chroma_stft.ht
ml

-[8] B. McFee, C. Raffel, D. Liang, D.P. Ellis, M. McVicar, E. Battenberg
and O. Nieto. "librosa: Audio and Music Signal Analysis in Python",
In Proceedings of the 14th Python in Science Conference, volume 8,
2015. doi: 10.25080/Majora-7b98e3ed-003

-[9] N., Marwan, M., Thiel, N.R. and Nowaczyk. “Cross Recurrence Plot
Based Synchronization of Time Series”. In: Nonlinear Processes in
Geophysics 9 (2002), 325-331. 2002.

-[10] ESC-50 veri kümesi, 2022. [Online].
https://github.com/karolpiczak/ESC-50

-[11] Çoklu Maskeleme işlemi dokümantasyonu, 2022. [Online].
https://www.tensorflow.org/io/tutorials/audio

-[12] Keras Kütüphanesi dokümantasyonu, 2022. [Online]. https://keras.io

-[13] ResNet50 sinir ağı modeli dokümantasyonu, 2022. [Online].
https://keras.io/api/applications/resnet/#resnet50-function

-[14] MobileNetV2 sinir ağı modeli dokümantasyonu, 2022. [Online].
https://keras.io/api/applications/mobilenet/#mobilenetv2-function

-[15] VGG16 ve VGG19 sinir ağı modelleri dokümantasyonu, 2022.
[Online]. https://keras.io/api/applications/vgg/#vgg16-function

-[16] Tensorflow kütüphanesi dokümantasyonu, 2022. [Online].
https://www.tensorflow.org/api_docs

-[17] Y., Aytar, C., Vondrick, A., Torralba. “SoundNet: Learning Sound
Representations from Unlabeled Video “. arXiv preprint arXiv :
1610.09001v1 [cs.CV] 2016

-[18] SoundNet kütüphanesinin Github İnternet Erişim Adresi, 2022,
[Online]. https://github.com/JarbasAl/soundnet

-[19] Altyapıda kullanılan çeşitli kütüphanelerin dokümantasyonları, 2022.
[Online]. https://pypi.org

-[20] DCASE (Detection and Classification of Acoustic Scenes and
Events) veri kümesi, 2022. [Online]. http://dcase.community

-[21] Kesit alınan bilgisayar oyununun oynanış videosu, 2022. [Online].
https://www.youtube.com/watch?v=d74REG039Dk

-[22] D.S., Park, W., Chan., Y., Zhang, C.-C., Chiu, B., Zoph, E.D., Cubuk
and Q.V., Le. “SpecAugment: A Simple Data Augmentation Method
for Automatic Speech Recognition”. arXiv preprint:
arXiv:1904.08779v3 [eess.AS]. 2019

-[23] J. You and J. Korhonen. "Attention Boosted Deep Networks For
Video Classification", 2020 IEEE International Conference on Image
Processing (ICIP), 2020, pp. 1761-1765. doi:
10.1109/ICIP40778.2020.9190996

-[24] Z., Zhang, S., Xu, S., Zhang, T., Qiao and S., Cao. "Learning
Attentive Representations for Environmental Sound Classification",
IEEE Access, vol. 7, pp. 130327-130339, 2019. doi:
10.1109/ACCESS.2019.2939495

-[25] B., Karasulu. “Kısıtlanmış Boltzmann makinesi ve farklı
sınıflandırıcılarla oluşturulan sınıflandırma iş hatlarının başarımının
değerlendirilmesi”, Bilişim Teknolojileri Dergisi, 11(3), 223-233,
2018. doi: 10.17671/gazibtd.370281

