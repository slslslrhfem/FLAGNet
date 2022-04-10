import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.pyplot as pp
from model import GAN_models
import random
from keras.preprocessing import sequence
from hyperparameter import Hyperparams as hp
from gan_util import generate_images
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from util import *

def matrix_cleaner(matrix,minimum_time):
  #matrix should be size of 24*24
  #make matrix's value of [0,1]  
  matrix=np.matrix(matrix)
  maximum_value=matrix.max()
  minimum_value=matrix.min()
  matrix=(matrix-minimum_value)/(maximum_value-minimum_value)
  flat=matrix.flatten()
  flat.sort()
  flat=flat.reshape((24*minimum_time,1))
  hundred_val=flat[-100]
  matrix_for_duration=np.where(matrix<0.05,0,matrix)
  matrix=np.where(matrix<hundred_val,0,matrix)
  matrix=np.where(matrix<1/3,0,matrix)
  
  return matrix,matrix_for_duration

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(image, footprint=neighborhood)==image

    #we create the mask of the background
    background = (image==0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    detected_peaks=np.where(detected_peaks==True,1,0)
    
    for i in range(len(detected_peaks[0])):
      for j in range(len(detected_peaks)):
        if(detected_peaks[j][i]==1):
          for k in range(i):
            if (image[j][i-k-1] != 0 and image[j][i-k-1] > image[j][i-k]):
              pass
            else:
              detected_peaks[j][i]=0
              detected_peaks[j][i-k-1]=1 #this code can decide note's position well. 
              #ex) 0.1 0.6 0.65 0.4 0.05... -> note starts at 2nd time with this condition.
              break
    return detected_peaks

def left_shifting(matrix,detected_peak,minimum_time):
  dots=[]
  times=[]
  for i in range(len(detected_peak[0])):
    for j in range(len(detected_peak)):
      if(detected_peak[j][i]==1):
        if(i%2!=0 and i%3!=0):#this code can shift Offbeats like 5/24, 7/24 to 4/24, 6/24, 
          detected_peak[j][i]=0
          detected_peak[j][i]=1 # i-1 -> Shift Offbeats, i -> Use Offbeats


  for i in range(len(detected_peak[0])):
    for j in range(len(detected_peak)):
      pos=[]
      a=0
      if(detected_peak[j][i]==1):  
        pos=[j,i]
      if(len(pos)!=0):
        dots.append(pos)
        times.append(i)
  dots_with_length=[]
  starting_points=[]
  durations=[]
  velocities=[]
  for i,position in enumerate(dots):
    velocity=matrix[1][position[0]][position[1]]
    if(position[1]!=0):

      if(velocity<matrix[1][position[0]][position[1]-1]):
        velocity=matrix[1][position[0]][position[1]-1] #for handling shifted notes
        #Note that matrix[1][position[0]][position[1]>matrix[1][position[0]][position[1]-1] for all none-shifted notes since
        #peak notes selected by local maximum neighboorhood.

      if velocity < 0.4:
        velocity=0.4
    velocities.append(velocity)
    length_val=0
    while True:
      if(length_val==0):
        starting_points.append([position[0],position[1]])
      else:
        if(position[1]+length_val>minimum_time-1):
          break
        if(detected_peak[position[0]][position[1]+length_val]!=0):
          break
      if(position[1]+length_val>minimum_time-1):
        break
      elif(matrix[1][position[0]][position[1]+length_val]!=0):
        dots_with_length.append([position[0],position[1]+length_val])
        times.append(position[1]+length_val)
      else:
        break
      length_val+=1

    while True:
      if(position[1]+length_val not in times and position[1]+length_val<minimum_time):
        dots_with_length.append([position[0],position[1]+length_val])
        times.append(position[1]+length_val)
        length_val+=1   
      else:
        break
    durations.append(length_val)
  result=np.zeros_like(matrix[0])
  for position in dots_with_length:
    result[position[0]][position[1]]=1
  return result,np.array(starting_points),np.array(durations),np.array(velocities)

C_chord=np.array([1,0,1,0,1,1,0,1,0,1,0,1]) # Formal C Major Notes. We can make another major scale by shifting this.
I_chord=np.array([1,0,0,0,1,0,0,1,0,0,0,0])
V_chord=np.array([0,0,1,0,0,0,0,1,0,0,0,1])
vi_chord=np.array([1,0,0,0,1,0,0,0,0,1,0,0])
IV_chord=np.array([1,0,0,0,0,1,0,0,0,1,0,0])
iii_chord=np.array([0,0,0,0,1,0,0,1,0,0,0,1])
ii_chord=np.array([0,0,1,0,0,1,0,0,0,1,0,0])
C_chords=np.array([I_chord,V_chord,vi_chord,IV_chord,iii_chord,ii_chord])

chords=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def chord_matching(shifted_matrix,chord,last_pitch,direction,with_chords=False):
  selected_chords=None
  before_pitch=last_pitch
  if (last_pitch<50):
    direction=1
  if (last_pitch>85):
    direction=0
  if chord not in chords:
    now_chord='C'
  now_chord=chord
  chord_diff=chords.index(now_chord)
  now_chord_list=np.roll(C_chord,chord_diff)#[1,0,1,0,1...]
  now_chords=[]
  for chord in C_chords:
    now_chords.append(np.roll(chord,chord_diff))
  pitch_set=[]
  for pitchs in shifted_matrix[1]:
    pitch_set.append(23-pitchs[0])
  pitch_set=pitch_set-pitch_set[0]#ex)0, 10, -1, 0 ,-4, -2
  if (direction==1):
    possible_set=[]
    for i in range(12):
      possible_set.append(pitch_set+i+last_pitch)
  else:
    possible_set=[]
    for i in range(12):
      possible_set.append(pitch_set-i+last_pitch)
  possible_set_score=[]
  if (with_chords):
    selected_chords=random.choice(now_chords)
    for sets in possible_set:
      score=0
      for pitchs in sets:
        score+=selected_chords[pitchs%12]
      possible_set_score.append(score)
    final_set=possible_set[np.argmax(possible_set_score)]
    return_val=[]
  else:
    for sets in possible_set:
      score=0
      for pitchs in sets:
        score+=now_chord_list[pitchs%12]
      possible_set_score.append(score)
    final_set=possible_set[np.argmax(possible_set_score)]
    return_val=[]
  for i,sets in enumerate(final_set):
    if (before_pitch-sets<-12):
      sets=sets-12
    if (before_pitch-sets>12):
      sets=sets+12               #For prevent pitch change above 1 Octave. If we want to use pitch change above 1 Octave, set 12 as 0.

    if(now_chord_list[sets%12]==1): #Use only matches with chord
      final=[]
      if (shifted_matrix[2][i]!=0):
        final.append(sets)#pitch
        final.append(shifted_matrix[1][i][1])
        final.append(shifted_matrix[2][i])
        final.append(shifted_matrix[3][i])
        return_val.append(final)
    else:
      final=[]
      if (shifted_matrix[2][i]!=0):
        final.append(sets-hp.pitch_shift)#pitch shifting for notes that not matches with chord scale. 
        final.append(shifted_matrix[1][i][1])
        final.append(shifted_matrix[2][i])
        final.append(shifted_matrix[3][i])
        return_val.append(final)
    before_pitch=sets
  return return_val, sets,selected_chords#final pitch

def generation_info(G,start_skill,length,chord,minimum_time,RNNmodel,updown_classifier,with_chords=False):
  mlb=MultiLabelBinarizer()
  labels=set_labels()
  mlb.fit(labels)
  for idx in range(hp.Label_num):
    if mlb.classes_[idx]=='no skills':
      no_skill_idx=idx
  skills=[]
  init_condition = np.zeros(128) # rel_pitch and rel_time vector
  skills.append(start_skill)
  infos=[]
  H=generate_images(G,1,start_skill,init_condition,init_condition)
  chord_sequence=[]
  shifted=left_shifting(matrix_cleaner(H,minimum_time),detect_peaks(matrix_cleaner(H,minimum_time)[0]),minimum_time)
  Chord_Match=chord_matching(shifted,chord,48,1,with_chords=True)
  infos.append(Chord_Match[0])
  chord_sequence.append(Chord_Match[2])
  last_pitch=Chord_Match[1]

  feature_seq=[]
  feature_seq.append(start_skill+1)
  feature_seq_pad=sequence.pad_sequences(np.array([feature_seq]),maxlen=20)
  prediction=RNNmodel.predict(feature_seq_pad)
  prediction[0][start_skill]=prediction[0][start_skill]/3
  prediction[0][idx]=0#no skills
  """
  if(minimum_time%3!=0):
    prediction[0][10]=0 # triplet skill, if minimum_time%3 is not 0, then generator can't handle any triplet notes.
  """
  updown_prediction=updown_classifier.predict(shifted[0].reshape(1,24,minimum_time,1))
  if(updown_prediction[0][0]>=updown_prediction[0][1]):
    updown_prediction=1
  else:
    updown_prediction=0
  prob=prediction/prediction.sum()
  next_skill=np.random.choice(
      np.arange(hp.Label_num),
      p=prob[0]
  )
  skills.append(next_skill)
  for i in range(length-1):
    time_condition=[]
    pitch_condition=[]
    for j,bars in enumerate(infos):
      for notes in bars:
        pitch_condition.append(notes[0]-last_pitch)
        time_condition.append((notes[1]-(len(infos)-j)*16)/4)
    
    time_condition = tf.keras.preprocessing.sequence.pad_sequences(np.array([time_condition]),maxlen=128)[0]
    pitch_condition = tf.keras.preprocessing.sequence.pad_sequences(np.array([pitch_condition]),maxlen=128)[0]

    #print(i+2,'th bar\'s generated skill is ',next_skill)
    H=generate_images(G,1,skills[int((i+1)%4)], time_condition, pitch_condition)
    shifted=left_shifting(matrix_cleaner(H,minimum_time),detect_peaks(matrix_cleaner(H,minimum_time)[0]),minimum_time)
    Chord_Match=chord_matching(shifted,chord,last_pitch,updown_prediction,with_chords=True)
    infos.append(Chord_Match[0])
    chord_sequence.append(Chord_Match[2])
    last_pitch=Chord_Match[1]
    feature_seq.append(next_skill+1)
    feature_seq_pad=sequence.pad_sequences(np.array([feature_seq]),maxlen=10)
    prediction=RNNmodel.predict(feature_seq_pad)
    prediction[0][next_skill]=prediction[0][next_skill]/3
    prediction[0]=prediction[0]+prediction.sum()/20 # for more variation.
    prediction[0][6]=0#6 is no skill
    updown_prediction=updown_classifier.predict(shifted[0].reshape(1,24,minimum_time,1))

    if(updown_prediction[0][0]>=updown_prediction[0][1]):
      updown_prediction=1
    else:
      updown_prediction=0
    prob=prediction/prediction.sum()
    next_skill=np.random.choice(
        np.arange(hp.Label_num),
        p=prob[0]
    )
    skills.append(next_skill)
  return infos,chord_sequence