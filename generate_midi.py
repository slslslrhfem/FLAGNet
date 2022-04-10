from model import GAN_models, RNN_models, classifier_models
from hyperparameter import Hyperparams as hp
from util import *
import tensorflow as tf
import keras
from midi_gen_util import *
import midiutil
from midiutil.MidiFile import MIDIFile


class MIDI_generator(object):
    def __init__(self):
        gan_model = GAN_models()
        self.G = gan_model.generator_model(hp.Minimum_time)
        self.G.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(hp.GAN_learning_rate, 0.5))
        self.G.load_weights("models/generator.h5")

        
        classifiers = classifier_models()
        self.updown_classifier=classifiers.make_updown_classifier()
        self.updown_classifier.compile(loss=keras.losses.CategoricalCrossentropy(
            from_logits=False, label_smoothing=0.1, 
            name='categorical_crossentropy'
        ), optimizer='adam', metrics=['accuracy'])
        self.updown_classifier.load_weights("models/updown.h5")

        self.label_classifier= classifiers.make_label_model()
        self.label_classifier.compile(loss=keras.losses.BinaryCrossentropy(
            from_logits=False, label_smoothing=0.1, 
            name='binary_crossentropy'
        ), optimizer='adam', metrics=['accuracy',recall,precision,f1score])
        self.label_classifier.load_weights("models/label_classifier.h5")

        RNN = RNN_models()
        self.RNN_model = RNN.make_RNNmodel()
        
        C_chord=np.array([1,0,1,0,1,1,0,1,0,1,0,1])
        I_chord=np.array([1,0,0,0,1,0,0,1,0,0,0,0])
        V_chord=np.array([0,0,1,0,0,0,0,1,0,0,0,1])
        vi_chord=np.array([1,0,0,0,1,0,0,0,0,1,0,0])
        IV_chord=np.array([1,0,0,0,0,1,0,0,0,1,0,0])
        iii_chord=np.array([0,0,0,0,1,0,0,1,0,0,0,1])
        ii_chord=np.array([0,0,1,0,0,1,0,0,0,1,0,0])
        self.C_chords=np.array([C_chord,I_chord,V_chord,vi_chord,IV_chord,iii_chord,ii_chord])

        self.chords=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    def gen_MIDI(self):
        mlb=MultiLabelBinarizer()
        labels=set_labels()
        mlb.fit(labels)
        chords = self.chords
        for start_skill in range(hp.Label_num):
            for chord in chords:
                self.RNN_model.load_weights("models/RNN.h5")
                final_list=generation_info(self.G,start_skill,hp.bar_length,chord,hp.Minimum_time,self.RNN_model,self.updown_classifier,with_chords=hp.with_chords)
                # create your MIDI object
                mf = MIDIFile(2)     # track number - 1
                track = 0   # the main track

                time = 0    # start at the beginning
                mf.addTrackName(track, time, "Sample Track")
                mf.addTempo(track, time, 120)#1bar per 2second.

                # add some notes
                channel = 0
                used_time=[]
                start_pitch=48
                for i,bars in enumerate(final_list[0]):
                    lowest_pitch=999
                    track = 0
                    for notes in bars:
                        pitch = notes[0]+12           # My implementation was 48 as C4, but this library think C4 as 60
                        if notes[1]%2==0:
                            time = notes[1]/(hp.Minimum_time/4)+i*4             # start on beat 0   0~15 / 4
                            duration = notes[2]/(hp.Minimum_time/4)
                        else: # This condition used for some testing..
                            time = (notes[1])/(hp.Minimum_time/4)+i*4
                            duration = (notes[2])/(hp.Minimum_time/4)         # 1 beat long

                        volume= int(notes[3]*100)
                        if (time not in used_time and duration!=0): #Delete not in used_time for polyphonic rhythm.
                            if volume<30:
                                volume=30
                            mf.addNote(track, channel, pitch, time, duration, volume)
                            used_time.append(time)
                            if pitch<lowest_pitch:
                                lowest_pitch=pitch
                    if (hp.with_chords is True):
                        track = 1
                        if (lowest_pitch!=999):
                            init_pitch=lowest_pitch
                        else:
                            init_pitch=start_pitch
                        chord_idx=np.where(final_list[1][i]==1)
                        pitch=0
                        iter=0
                        pitch_in_chord=[]
                        while (pitch<init_pitch):
                            for idx in chord_idx[0]:
                                pitch=idx+12*iter
                                if (pitch>=init_pitch):
                                    break
                                pitch_in_chord.append(pitch)
                            iter+=1

                        pitch=pitch_in_chord[len(pitch_in_chord)-1]
                        time = i*4
                        duration=4.0
                        volume=50
                        mf.addNote(track,channel,pitch,time,duration,volume)
                        pitch=pitch_in_chord[len(pitch_in_chord)-2]
                        time = i*4
                        duration=4.0
                        volume=50
                        mf.addNote(track,channel,pitch,time,duration,volume)
                        pitch=pitch_in_chord[len(pitch_in_chord)-3]
                        time = i*4
                        duration=4.0
                        volume=50
                        mf.addNote(track,channel,pitch,time,duration,volume)
                        start_pitch=init_pitch
                with open("MIDI_result/"+mlb.classes_[start_skill]+chord+".mid", 'wb') as outf:
                    mf.writeFile(outf)
                    print(mlb.classes_[start_skill]+chord+"  generate done!")
                    #print('info is \n',final_list)