import sys
import numpy as np
from generate_midi import MIDI_generator
from model import GAN_models, RNN_models, classifier_models
from preprocess import preprocessing
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

def main():
    # Expected Running Code 
    """
    python main.py preprocessing
    python training
    python generate_midi

    Note that processed data and training parameters already in Github Project folder, So you can use only  "python generate_midi" code for generate samples.
    preprocessing and training may spend long time.
    """

    if sys.argv[1] == 'train_classifier': # this learn 2 classifier model and save two model.
        classifiers = classifier_models()
        classifiers.label_classifier_train()
        classifiers.updown_classifier_train()
    elif sys.argv[1] == 'preprocessing':
        preprocessing()
    elif sys.argv[1] == 'train_gan':
        gan_models = GAN_models()
        gan_models.training_gan()
        pass
    elif sys.argv[1] == 'generate_midi':
        generator = MIDI_generator()
        generator.gen_MIDI()
        pass
    elif sys.argv[1] == 'train_rnn':
        rnn_models = RNN_models()
        rnn_models.RNN_train()
    elif sys.argv[1] == 'training':
        classifiers = classifier_models()
        classifiers.label_classifier_train()
        classifiers.updown_classifier_train()
        gan_models = GAN_models()
        gan_models.training_gan()
        rnn_models = RNN_models()
        rnn_models.RNN_train()
    
    else:
        all_labels = np.load('preprocessing/all_labels.npy', allow_pickle=True)
        tot_dict={}
        for label in all_labels:
            for skills in label:
                if skills not in tot_dict:
                    tot_dict[skills] = 1
                else:
                    tot_dict[skills] += 1
        print(tot_dict)
    

if __name__=="__main__":
    main()