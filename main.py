import keras
import os
import json
import pandas
import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import copy
import pretty_midi
import mido
from tqdm import tqdm
from model import f1score,precision,make_classifier,make_model,recall
from utils import get_meta,csv_to_array,midi_to_array,bar_to_matrix2
from feature_defining import bar_to_contour, contour_to_label
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt


midifilenames=sorted(os.listdir('PPDD-Sep2018_sym_mono_small/PPDD-Sep2018_sym_mono_small/prime_midi'))
jsonfilenames=sorted(os.listdir('PPDD-Sep2018_sym_mono_small/PPDD-Sep2018_sym_mono_small/descriptor'))
csvfilenames=sorted(os.listdir('PPDD-Sep2018_sym_mono_small/PPDD-Sep2018_sym_mono_small/prime_csv'))
midilist=[]
csvlist=[]
jsonlist=[]
prettymidilist=[]

for filenames in tqdm(midifilenames,position=0):
  midi_path='PPDD-Sep2018_sym_mono_small/PPDD-Sep2018_sym_mono_small/prime_midi/'+filenames
  mid = mido.MidiFile(midi_path, clip=True)
  midilist.append(mid)
  prettymid=pretty_midi.PrettyMIDI(midi_path)
  prettymidilist.append(prettymid)
for filenames in tqdm(csvfilenames,position=0):
  csv_path='PPDD-Sep2018_sym_mono_small/PPDD-Sep2018_sym_mono_small/prime_csv/'+filenames
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

#   dot 기반의 matrix는 사용하지 않는다.

#bar_list to bar_matrix_list
#bar_matrix_list=copy.deepcopy(bar_list)
bar_matrix_list2=copy.deepcopy(bar_list)
for i,songs in enumerate(bar_matrix_list2):
  for j,bar in enumerate(songs):
    #print(one_bar_number_list[i],starting_number_list[i])
    #matrix=bar_to_matrix1(bar,one_bar_number_list[i],starting_number_list[i],j)
    # bar_matrix_list[i][j]=matrix
    matrix2=bar_to_matrix2(bar,one_bar_number_list[i],starting_number_list[i],j)
    bar_matrix_list2[i][j] = matrix2

#아래의 코드는 bar를 plot할때 사용. 필요하면 주석을 풀면 된다.
#plot_bar(bar_matrix_list2[0][0])

bar_contour_list=copy.deepcopy(bar_list)
bar_label_list=copy.deepcopy(bar_contour_list)
for i,songs in enumerate(bar_list):
  for j,bar in enumerate(songs):
    contour=bar_to_contour(bar,one_bar_number_list[i],starting_number_list[i],j)
    bar_contour_list[i][j]=contour
    label=contour_to_label(contour)
    bar_label_list[i][j]=label

"""
여기부터 Tensorflow 기반의 model 구현
"""
all_matrix=[]
all_labels=[]
num_data=len(all_matrix)
for songs in bar_label_list:
  for label in songs:
    label=np.array(label)
    all_labels.append(label)
bar_label_list=[]#램 터짐
for songs in bar_matrix_list2:
  for matrix in songs:
    matrix=matrix.reshape(112,96,1)
    all_matrix.append(matrix)
bar_matrix_list2=[]#이거도

train_matrix=np.array(all_matrix[:1500])
train_label=np.array(all_labels[:1500])
valid_matrix=np.array(all_matrix[1500:1700])
valid_label=np.array(all_labels[1500:1700])
test_matrix=np.array(all_matrix[1700:])
test_label=np.array(all_labels[1700:])
mlb=MultiLabelBinarizer()
mlb.fit([['no skills','resting','repeating','up_steping','down_steping','up_leaping','down_leaping','steping_twisting','leaping_twisting','fast_rhythm','One_rhythm','triplet','staccato','continuing_rhythm']])
train_label2=mlb.transform(train_label)
valid_label2=mlb.transform(valid_label)
test_label2=mlb.transform(test_label)

model_path = 'models/deeperppddbest.h5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy',
                                verbose=1, save_best_only=True)
callbacks = [cb_checkpoint]

classifier=make_model()
classifier.compile(loss=keras.losses.BinaryCrossentropy(
      from_logits=False, label_smoothing=0.1,
      name='binary_crossentropy'
  ), optimizer='adam', metrics=['accuracy',recall,precision,f1score])
classifier.fit(
      train_matrix,train_label2,batch_size=32,
      epochs=50,
      validation_data=(valid_matrix,valid_label2)
  ,verbose=2, callbacks=callbacks
)

# testing

testresult=classifier.predict(test_matrix)

classnum={}
testnum={}
resultmat=[]
for i in range(len(testresult)):
  eval_result=[0 for i in range(14)]
  class_num=np.count_nonzero(test_label2[i]==1)
  classidx=(-testresult[i]).argsort()[:class_num]
  for j in classidx:
    eval_result[j]=1
  resultmat.append(eval_result)
resultmat=np.array(resultmat)
testidx=mlb.inverse_transform(resultmat)

classidx=mlb.inverse_transform(test_label2)
for i in range(len(testidx)):
  for classes in classidx[i]:
    if (classes not in classnum):
      classnum[classes]=1
    else:
      classnum[classes]+=1
  for classes in testidx[i]:
    if (classes not in testnum):
      testnum[classes]=1
    else:
      testnum[classes]+=1
print(classnum, testnum) # for check balancing