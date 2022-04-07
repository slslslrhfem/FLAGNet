import pandas as pd
from keras import backend as K
import tensorflow as tf
import music21
import numpy as np
import copy
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import scipy as sp
import scipy.ndimage

def get_meta():
  file=pd.read_excel('POP909/index.xlsx', engine='openpyxl')
  return file

def extract_notes(midi_part):
    parent_element = []
    ret = []
    z=[]
    vel=[]
    for nt in midi_part.flat.notes:        
        if isinstance(nt, music21.note.Note):
            ret.append(max(0.0, nt.pitch.ps))
            parent_element.append(nt)
            z.append(max(nt.duration.quarterLength,0.125))
            vel.append(nt.volume.velocity)
        elif isinstance(nt, music21.chord.Chord):
            for pitch in nt.pitches:
                ret.append(max(0.0, pitch.ps))
                parent_element.append(nt)
                z.append(max(nt.duration.quarterLength,0.25))
                vel.append(nt.volume.velocity)
    x=[n.offset for n in parent_element]
    return x, ret, parent_element,z,vel

def get_ts(index_number,meta_data):
  TS=[0,0]
  TS[0]=meta_data.iloc[index_number,4]
  TS[1]=meta_data.iloc[index_number,5]
  return TS

def numbershifting(number,time):
  a=time
  while True:
    if number<a:
      a+=time
    else:
      break
  return a

def double(lst):
    return [i*2 for i in lst]
    
def nearest_time(time,minimum_size):
  #Given minimum time, there can be outliars. Shift for this. ex)32th notes in minimum unit 16th notes.
  num_to_multiply=time/minimum_size
  num_to_multiply=int(num_to_multiply)
  left_time=num_to_multiply*minimum_size
  right_time=left_time+minimum_size
  if (time-left_time>=right_time-time):
    return right_time
  return left_time

# Since label which has both 'up_steping' and 'down_steping' is impossible, setting label against with these impossible cases to improve classifier's performance.
def set_labels():
  labels=[]
  label_tuple=[]
  skills_pitch=['repeating','up_steping','down_steping','up_leaping','down_leaping','steping_twisting','leaping_twisting','dummy']
  skills_timing=['fast_rhythm','dummy']
  skills_triplet=['triplet','dummy']
  skills_one_rhythm=['One_rhythm','dummy']
  skills_staccato=['staccato','continuing_rhythm','dummy']
  for pitch in skills_pitch:
    for timing in skills_timing:
      for triplet in skills_triplet:
        for one_rhythm in skills_one_rhythm:
          for staccato in skills_staccato:
            label_tuple=[]
            if pitch != 'dummy':
              label_tuple.append(pitch)
            if timing != 'dummy':
              label_tuple.append(timing)
            if triplet != 'dummy':
              label_tuple.append(triplet)
            if one_rhythm != 'dummy':
              label_tuple.append(one_rhythm)
            if staccato != 'dummy':
              label_tuple.append(staccato)
            if len(label_tuple)==0:
              label_tuple.append('no skills') # no skills label is used for training classifier and generator, but not used for real generation.
            label_tuple=tuple(label_tuple)
            
            labels.append(label_tuple)
  
  return labels


def recall(y_target, y_pred):
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 
    count_true_positive_false_negative = K.sum(y_target_yn)
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 
    count_true_positive_false_positive = K.sum(y_pred_yn)
    # Precision = (True Positive) / (True Positive + False Positive)
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    
    # return a single tensor value
    return _f1score

def get_tag_results(testresult,test_label2):  
  classnum={}
  testnum={}
  resultmat=[]
  bestmat=[]
  for i in range(len(testresult)):
    eval_result=[0 for i in range(13)]
    best_result=[0 for i in range(13)]
    class_num=np.count_nonzero(test_label2[i]==1)+1
    classidx=(-testresult[i]).argsort()[:class_num]
    for k,j in enumerate(classidx):
      if (k==0):
        best_result[j]=1
      eval_result[j]=1
    resultmat.append(eval_result)
    bestmat.append(best_result)
    test_result2=copy.deepcopy(testresult)
    test_result2[np.where(test_result2>0.30)]=1
    test_result2[np.where(test_result2<=0.30)]=0
  resultmat=np.array(resultmat)
  bestmat=np.array(bestmat)
  mlb=MultiLabelBinarizer()
  labels=set_labels()
  mlb.fit(labels)
  testidx=mlb.inverse_transform(resultmat)
  classidx=mlb.inverse_transform(test_label2)
  testidx2=mlb.inverse_transform(test_result2)
  bestidx=mlb.inverse_transform(bestmat)
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
  print(classnum, testnum)
  return bestidx, testidx2, testidx, classidx

def get_best_results(testresult,test_label2):  #for optimizing get_tag_results function
  classnum={}
  testnum={}
  resultmat=[]
  bestmat=[]
  mlb=MultiLabelBinarizer()
  labels=set_labels()
  mlb.fit(labels)
  
  for i in range(len(testresult)):
    best_result=[0 for i in range(13)]
    class_num=np.count_nonzero(test_label2[i]==1)+1
    classidx=(-testresult[i]).argsort()[:class_num]
    for k,j in enumerate(classidx):
      if (k==0):
        best_result[j]=1
    bestmat.append(best_result)
  bestmat=np.array(bestmat)
  bestidx=mlb.inverse_transform(bestmat)
  return bestidx

def blur_image(matrix):
  sigma_y = 0.5
  sigma_x = 0.5
  inputmat=matrix
  # Apply gaussian filter
  sigma = [sigma_y, sigma_x]
  y = sp.ndimage.filters.gaussian_filter(inputmat, sigma, mode='constant')
  return y