import matplotlib
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from keras import utils as keras_utils
from Decoding import generation_info
from keras import datasets
from swiss_army_tensorboard import tfboard_loggers
from RNNmodel import RNN_training
import keras
import os
import pandas
import numpy as np
from keras import optimizers
from midiutil.MidiFile import MIDIFile
from cdcgan import cdcgan_models, cdcgan_utils
from keras.callbacks import ModelCheckpoint
import copy
import pretty_midi
import mido
from tqdm import tqdm
from model import f1score,precision,make_classifier,make_model,recall
from utils import get_meta,csv_to_array,midi_to_array,bar_to_matrix3,set_labels,get_best_results,blur_image
from feature_defining import bar_to_contour, contour_to_label
from sklearn.preprocessing import MultiLabelBinarizer
from plotting import plot_recall_f1score, plot_val_loss
import cv2

midifilenames=sorted(os.listdir('PPDD-Sep2018_sym_mono_large/prime_midi'))
jsonfilenames=sorted(os.listdir('PPDD-Sep2018_sym_mono_large/descriptor'))
csvfilenames=sorted(os.listdir('PPDD-Sep2018_sym_mono_large/prime_csv'))

midilist=[]
csvlist=[]
jsonlist=[]
prettymidilist=[]

for filenames in tqdm(midifilenames,position=0):
  midi_path='PPDD-Sep2018_sym_mono_large/prime_midi/'+filenames
  mid = mido.MidiFile(midi_path, clip=True)
  midilist.append(mid)
  prettymid=pretty_midi.PrettyMIDI(midi_path)
  prettymidilist.append(prettymid)
for filenames in tqdm(csvfilenames,position=0):
  csv_path='PPDD-Sep2018_sym_mono_large/prime_csv/'+filenames
  csv = pandas.read_csv(csv_path)
  csvlist.append(csv)
for filenames in tqdm(jsonfilenames,position=0):
  jsonlist.append(get_meta(filenames))

# list of filenames are ready.

bar_list, one_bar_number_list, starting_number_list = midi_to_array(prettymidilist,jsonlist)#can use csv_to_array(csvlist,jsonlist)

"""
bar_list[i][j][k] means ith music, jth bar kth note. bar_list contains note information
one_bar_number_list[i] means ith music's music time Signature information. if song has 3/4 signature, it has '3' value.
starting_number_list[i] means ith music's first note starting time.
"""

Minimum_time=24
bar_matrix_list3=copy.deepcopy(bar_list)
for i,songs in enumerate(bar_matrix_list3):
  for j,bar in enumerate(songs):
    matrix3=bar_to_matrix3(bar,one_bar_number_list[i],starting_number_list[i],j,Minimum_time)
    bar_matrix_list3[i][j] = matrix3

bar_updown_list=copy.deepcopy(bar_list)
for i,songs in enumerate(bar_list):
  for j,bar in enumerate(songs):
    if (j==len(songs)-1):
      updown_label='final'
    elif(len(bar_list[i][j])==0 or len(bar_list[i][j+1])==0):
      updown_label='meanless'
    else:
      if(bar_list[i][j][len(bar_list[i][j])-1][1]<=bar_list[i][j+1][0][1]):
        updown_label='up'
      else:
        updown_label='down'
    bar_updown_list[i][j]=updown_label

bar_contour_list=copy.deepcopy(bar_list)
bar_label_list=copy.deepcopy(bar_contour_list)
for i,songs in enumerate(bar_list):
  for j,bar in enumerate(songs):
    contour=bar_to_contour(bar,one_bar_number_list[i],starting_number_list[i],j)
    bar_contour_list[i][j]=contour
    label=contour_to_label(contour)
    bar_label_list[i][j]=label

"""
Model implementation with tensorflow
"""
all_matrix=[]
all_labels=[]
all_updown_labels=[]
for songs in bar_label_list:
  for label in songs:
    label=np.array(label)
    all_labels.append(label)
for songs in bar_matrix_list3:
  for matrix in songs:
    matrix=matrix.reshape(24,Minimum_time,1)
    all_matrix.append(matrix)
for songs in bar_updown_list:
  for label in songs:
    all_updown_labels.append(label)
updownle=LabelBinarizer()
updownle.fit(['up','down','final','meanless'])
updown_label=updownle.transform(np.array(all_updown_labels))
tot_len=len(all_matrix)
train_matrix=np.array(all_matrix[:int(len(all_matrix)*0.85)])
train_label=np.array(all_labels[:int(len(all_matrix)*0.85)])
valid_matrix=np.array(all_matrix[int(len(all_matrix)*0.85):int(len(all_matrix)*0.95)])
valid_label=np.array(all_labels[int(len(all_matrix)*0.85):int(len(all_matrix)*0.95)])
test_matrix=np.array(all_matrix[int(len(all_matrix)*0.95):])
test_label=np.array(all_labels[int(len(all_matrix)*0.95):])
mlb=MultiLabelBinarizer()
labels=set_labels()
mlb.fit(labels)
train_label2=mlb.transform(train_label)
valid_label2=mlb.transform(valid_label)
test_label2=mlb.transform(test_label)
train_updown_label=updown_label[:int(0.85*tot_len)]
valid_updown_label=updown_label[int(0.85*tot_len):int(0.95*tot_len)]
test_updown_label=updown_label[int(0.95*tot_len):]

model_path = 'models/deeperppddbest.h5'
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy',
                                verbose=1, save_best_only=True)
callbacks = [cb_checkpoint]

model_path = 'models/updown.h5'
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy',
                                verbose=1, save_best_only=True)
updown_callbacks=[cb_checkpoint]

classifier=make_model(Minimum_time)
classifier.compile(loss=keras.losses.BinaryCrossentropy(
      from_logits=False, label_smoothing=0.1,
      name='binary_crossentropy'
  ), optimizer='adam', metrics=['accuracy',recall,precision,f1score])
hist=classifier.fit(
      train_matrix,train_label2,batch_size=256,
      epochs=50,
      validation_data=(valid_matrix,valid_label2)
  ,verbose=2, callbacks=callbacks
)
classifier.load_weights('models/deeperppddbest.h5')

#plotting
plot_val_loss(hist)
plot_recall_f1score(hist)

updown_classifier=make_classifier(Minimum_time)
updown_classifier.compile(loss=keras.losses.BinaryCrossentropy(
      from_logits=False,
      name='binary_crossentorpy',
  ), optimizer='adam', metrics=['accuracy'])

updown_hist=updown_classifier.fit(
      train_matrix,train_updown_label,batch_size=128,
      epochs=50,
      validation_data=(valid_matrix,valid_updown_label),
      callbacks=updown_callbacks,
  )
updown_classifier.load_weights('models/' + 'updown.h5')

testresult=classifier.predict(test_matrix)

BATCH_SIZE = 256
EPOCHS = 100

trainX=train_matrix.reshape((int(len(all_matrix)*0.85),24,Minimum_time))
test_result=classifier.predict(train_matrix)
trainy=np.array(get_best_results(test_result,train_label2)).reshape((int(len(all_matrix)*0.85),))

encoder = LabelEncoder()

# X_train데이터를 이용 피팅하고 라벨숫자로 변환한다
encoder.fit(trainy)
trainy = encoder.transform(trainy)
trainy=np.array(trainy)

#사진에 filter를 적용한다.
blur_trainX=copy.deepcopy(trainX)
for i,matrix in enumerate(trainX):
  blur_trainX[i]=blur_image(matrix)
trainX=blur_trainX

X_train=trainX
y_train=trainy
X_train = cdcgan_utils.transform_images(X_train)
X_train = X_train[:, :, :, None]
# Create the models
y_train = keras_utils.to_categorical(y_train, 13)

print("Generator:")
G = cdcgan_models.generator_model(Minimum_time)
G.summary()

print("Discriminator:")
D = cdcgan_models.discriminator_model(Minimum_time)
D.summary()

print("Combined:")
GD = cdcgan_models.generator_containing_discriminator(G, D)
GD.summary()

optimizer = optimizers.Adam(0.0002, 0.5)
G.compile(loss='binary_crossentropy', optimizer=optimizer)
GD.compile(loss='binary_crossentropy', optimizer=optimizer)
D.trainable = True
D.compile(loss='binary_crossentropy', optimizer=optimizer)

# Setup Tensorboard loggers

tfboard_loggers.TFBoardModelGraphLogger.log_graph("cdcgan/GANresult/models/logs", K.get_session())
loss_logger = tfboard_loggers.TFBoardScalarLogger("cdcgan/GANresult/models/logs/loss")
image_logger = tfboard_loggers.TFBoardImageLogger("cdcgan/GANresult/models/logs/generated_images")

# Model Training

iteration = 0

nb_of_iterations_per_epoch = int(X_train.shape[0] / BATCH_SIZE)
print("Number of iterations per epoch: {0}".format(nb_of_iterations_per_epoch))

for epoch in range(EPOCHS):
    pbar = tqdm(desc="Epoch: {0}".format(epoch), total=X_train.shape[0])

    g_losses_for_epoch = []
    d_losses_for_epoch = []

    for i in range(nb_of_iterations_per_epoch):
        noise = cdcgan_utils.generate_noise((BATCH_SIZE, 13))

        image_batch = X_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        label_batch = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        generated_images = G.predict([noise, label_batch], verbose=0)

        if i % 20 == 0:
            image_grid = cdcgan_utils.generate_mnist_image_grid(G,
                                                                title="Epoch {0}, iteration {1}".format(epoch,
                                                                                                        iteration))
            cdcgan_utils.save_generated_image(image_grid, epoch, i, "cdcgan/GANresult/images/generated_mnist_images_per_iteration")
            image_logger.log_images("generated_mnist_images_per_iteration", [image_grid], iteration)

        X = np.concatenate((image_batch, generated_images))
        y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
        label_batches_for_discriminator = np.concatenate((label_batch, label_batch))

        D_loss = D.train_on_batch([X, label_batches_for_discriminator], y)
        d_losses_for_epoch.append(D_loss)
        loss_logger.log_scalar("discriminator_loss", D_loss, iteration)

        noise = cdcgan_utils.generate_noise((BATCH_SIZE, 13))
        D.trainable = False
        G_loss = GD.train_on_batch([noise, label_batch], [1] * BATCH_SIZE)
        D.trainable = True
        g_losses_for_epoch.append(G_loss)
        loss_logger.log_scalar("generator_loss", G_loss, iteration)

        pbar.update(BATCH_SIZE)

        iteration += 1

    # Save a generated image for every epoch
    image_grid = cdcgan_utils.generate_mnist_image_grid(G, title="Epoch {0}".format(epoch))
    cdcgan_utils.save_generated_image(image_grid, epoch, 0, "cdcgan/GANresult/images/generated_mnist_images_per_epoch")
    image_logger.log_images("generated_mnist_images_per_epoch", [image_grid], epoch)

    pbar.close()
    print("D loss: {0}, G loss: {1}".format(np.mean(d_losses_for_epoch), np.mean(g_losses_for_epoch)))

    G.save_weights("cdcgan/GANresult/models/weights/generator.h5")
    D.save_weights("cdcgan/GANresult/models/weights/discriminator.h5")

with_chords=True
#RNN implementation
RNNmodel=RNN_training(classifier,bar_matrix_list3,Minimum_time)

chords=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
for start_skill in range(13):
  for chord in chords:
    final_list=generation_info(start_skill,16,chord,Minimum_time,RNNmodel,updown_classifier,with_chords=with_chords)
    # create your MIDI object
    mf = MIDIFile(1)     # only 1 track
    track = 0   # the only track

    time = 0    # start at the beginning
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, 120)#2초에 1bar

    # add some notes
    channel = 0
    used_time=[]
    for i,bars in enumerate(final_list[0]):
      lowest_pitch=999
      for notes in bars:
        pitch = notes[0]+12           # C4 (middle C) 48이 C4인 내구현에 비해 여기는 60이 C4이다.
        time = notes[1]/(Minimum_time/4)+i*4             # start on beat 0
        duration = notes[2]/(Minimum_time/4)         # 1 beat long
        volume= int(notes[3]*100)
        if (time not in used_time and duration!=0): #not in used_time 빼면 화음도 들간다.
          mf.addNote(track, channel, pitch, time, duration, volume)
          used_time.append(time)
          if pitch<lowest_pitch:
            lowest_pitch=pitch
      if (with_chords is True):
        init_pitch=lowest_pitch
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
    with open("/content/drive/My Drive/MARG/PPDDlist/MIDI_result/output"+mlb.classes_[start_skill]+chord+".mid", 'wb') as outf:
      mf.writeFile(outf)
      print(mlb.classes_[start_skill]+chord+"  generate done!")