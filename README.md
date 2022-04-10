# FLAGNet

## Description

**Feature Label based Automatic Generational Network for Symbolic Music Generation.**

Piano-roll based processing with relational pitch and relational time, and Feature label conditioned GAN with primining notes conditioning which has pix2pix structure. 

dcGAN model lots of help from https://github.com/gaborvecsei/CDCGAN-Keras/tree/master/cdcgan.

There are some cherry-picked sample in sample folder, and whole results in MIDI_Result. Results in MIDI_Result should be changed if I use this model and commit it!

## How to use

1. Prepare Python=3.6 Environments. ( Note that This Project had some issue in model for Python=3.9 Environments )

If you use conda Virtual Environments, then..

    conda create -n flagnet env python=3.6
    conda activate flagnet

2. Install prerequisties.

Just type in terminal, 

    pip install -r requirements.txt

Note that this requirements has "tensorflow", and it may requires CUDA GPU setting for running codes. 

3. Setting Hyperparameters(Optional)
There is some hyperparameter setting or musical decoding condition(Minimum notes unit, use notes in scale or not, etc..) parameters in **hyperparameter.py**.

4. Run Main Codes

main.py code gets one argument that leads what to do,

    python main.py preprocess
    
There is 909 midi files in data but I used 300 file only because of CPU killing issues.

But if you want to use whole data, then modify some codes in **preprocess.py**, line 19 and 35, *pop909namelist[:300]* to *pop909namelist*. Then,

    python main.py training
  
It trains label/direction classifier, image generation dcGAN, and label sequence RNN.
You can also train each models with

    python main.py train_classifier
    python main.py train_gan
    python main.py train_rnn

Note that above 2 method works exactly same!

And then,

    python main.py generate_midi
    
    
This model should generated MIDIs, with name [First bar's skill label][chord scale].mid in midi_result folder.
 
colab notebook (Full code & Tutorial) will be added!
