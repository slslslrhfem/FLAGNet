# Implementation of FLAGNet
Paper here : https://openreview.net/pdf?id=K_ETaDx3Iv.

Please Use
tensorflow==1.15.0
Keras==2.3.1
Keras_Applications==1.0.8
and mido, prettymidi, and so on..

If you want to apply this code for own midi-data, or understanding flow of code, then I recommend to use ipynb file with colab, with GPU mode.
Download this github folder with name 'FLAGNet', place the folder at base folder of google drive and Mount Your google drive.

(*It can be hard to upload all the datasets, So I recommend you to load file with PPDDlist/(something)list.txt.

PPDDlist folder is here :  https://drive.google.com/file/d/1Bwq6SHUPhLvurcy6pf1GTAUjV4ssztMM/view?usp=sharing
I use drive sharing because of data size issue. Please move PPDDlist folder in FLAGNet folder in google drive and follow ipynb code.
)

with .py files, just run main.py. 

GAN model is from https://github.com/gaborvecsei/CDCGAN-Keras/tree/master/cdcgan.
