import sys
from generate_midi import MIDI_generator
from model import GAN_models, RNN_models, classifier_models
from preprocess import preprocessing
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

def main():
    print(tf.__version__)
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
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
    else:
        print(sys.argv[1])
    

if __name__=="__main__":
    main()