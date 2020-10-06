import numpy as np
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from keras.preprocessing import sequence
from cdcgan import cdcgan_utils
import random
import matplotlib.pyplot as pp
C_chord=np.array([1,0,1,0,1,1,0,1,0,1,0,1])
I_chord=np.array([1,0,0,0,1,0,0,1,0,0,0,0])
V_chord=np.array([0,0,1,0,0,0,0,1,0,0,0,1])
vi_chord=np.array([1,0,0,0,1,0,0,0,0,1,0,0])
IV_chord=np.array([1,0,0,0,0,1,0,0,0,1,0,0])
iii_chord=np.array([0,0,0,0,1,0,0,1,0,0,0,1])
ii_chord=np.array([0,0,1,0,0,1,0,0,0,1,0,0])
C_chords=np.array([I_chord,V_chord,vi_chord,IV_chord,iii_chord,ii_chord])
chords=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def matrix_cleaner(matrix):
  #matrix should be size of 24*24
  #make matrix's value of [0,1]
  matrix=np.matrix(matrix)
  maximum_value=matrix.max()
  minimum_value=matrix.min()
  matrix=(matrix-minimum_value)/(maximum_value-minimum_value)
  flat=matrix.flatten()
  flat.sort()
  flat=flat.reshape((576,1))
  hundred_val=flat[-70]
  matrix=np.where(matrix<hundred_val,0,matrix)
  matrix=np.where(matrix<0.35,0,matrix)
  return matrix




def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to
    #successfully subtract it form local_max, otherwise a line will
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks,
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    detected_peaks=np.where(detected_peaks==True,1,0)

    return detected_peaks

def left_shifting(matrix,detected_peak,minimum_time):
  #input은 detected_peak가 들어와야한다.
  dots=[]
  times=[]
  """
  for i in range(len(detected_peak[0])):
    for j in range(len(detected_peak)):
      if(detected_peak[j][i]==1):
        if(i%2!=0 and i%3!=0):#this code can shift Offbeats like 5/24, 7/24 to 4/24, 6/24, 
          detected_peak[j][i]=0
          detected_peak[j][i-1]=1
  """
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

def chord_matching(shifted_matrix,chord,last_pitch,direction,with_chords=False):
  #input은 left_shifting의 output을 그대로 넣고, chord(C,C#,D,D#....,B), 마지막 리듬의 pitch를 숫자로, 그리고 진행 방향을 direction(1,0)으로 받는다.
  #output은 MIDI기반의 decoding이 되도록 정보 기반으로, 그리고 last_pitch까지 넣어준다.
  #또한 출력 그림과 상관없이, 그냥 Note Pitch를 그냥 Matrix에 넣어준다.
  selected_chords=None
  before_matrix=shifted_matrix[0]
  before_pitch=last_pitch
  if (last_pitch<50):
    direction=1
  if (last_pitch>85):
    direction=0
  if chord not in chords:
    now_chord='C'
  now_chord=chord
  chord_diff=chords.index(now_chord)
  now_chord_list=np.roll(C_chord,chord_diff)#[1,0,1,0,1...] 이런 set인데 여기에 맞추면 함수에 넣은 chord에 맞게 된다.
  #나중에 그냥 chord뿐만 아니라 화음의 종류까지 고민하게 된다면 이걸 잘 쓰면 된다.
  now_chords=[]
  for chord in C_chords:
    now_chords.append(np.roll(chord,chord_diff))
  pitch_set=[]
  for pitchs in shifted_matrix[1]:
    pitch_set.append(23-pitchs[0])
  pitch_set=pitch_set-pitch_set[0]#이러면 0, 10, -1, 0 ,-4, -2, 이렇게 들간다. 상대적인 위치를 다루는게 쉽다.
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
      sets=sets+12               #For prevent pitch change above 1 Octave.
    if(now_chord_list[sets%12]==1): #Use only
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
        final.append(sets)#pitch shifting for notes that not matches with chord scale. If you want to use these type of notes, just remove -1 in sets-1.
        final.append(shifted_matrix[1][i][1])
        final.append(shifted_matrix[2][i])
        final.append(shifted_matrix[3][i])
        return_val.append(final)
    before_pitch=sets
  return return_val, sets,selected_chords#final pitch
def generation_info(start_skill,length,chord,minimum_time,RNNmodel,updown_classifier,with_chords=False):
  infos=[]
  H=cdcgan_utils.generate_images(G,1,start_skill)
  chord_sequence=[]
  shifted=left_shifting(matrix_cleaner(H,minimum_time),detect_peaks(matrix_cleaner(H,minimum_time)[0]),minimum_time)
  Chord_Match=chord_matching(shifted,chord,48,1,with_chords=True)
  infos.append(Chord_Match[0]) # 48은 last_pitch정보, 1은 updown precition, 즉 direction 관련 정보다.
  chord_sequence.append(Chord_Match[2])
  last_pitch=Chord_Match[1]

  feature_seq=[]
  feature_seq.append(start_skill+1)#RNN Predict Pad 전에 1을 더해야한다.
  feature_seq_pad=sequence.pad_sequences(np.array([feature_seq]),maxlen=10)
  prediction=RNNmodel.predict(feature_seq_pad)
  prediction[0][start_skill]=prediction[0][start_skill]/3
  prediction[0][6]=0#no skills
  if(minimum_time%3!=0):
    prediction[0][10]=0 # triplet skill, if minimum_time%3 is not 0, then generator can't handle any triplet notes.
  updown_prediction=updown_classifier.predict(shifted[0].reshape(1,24,minimum_time,1))
  if(updown_prediction[0][0]>=updown_prediction[0][1]):
    updown_prediction=1
  else:
    updown_prediction=0
  prob=prediction/prediction.sum()
  next_skill=np.random.choice(
      np.arange(13),
      p=prob[0]
  )
  for i in range(length-1):
    #print(i+2,'th bar\'s generated skill is ',next_skill)
    H=cdcgan_utils.generate_images(G,1,next_skill)
    shifted=left_shifting(matrix_cleaner(H,minimum_time),detect_peaks(matrix_cleaner(H,minimum_time)[0]),minimum_time)
    Chord_Match=chord_matching(shifted,chord,last_pitch,updown_prediction,with_chords=True)
    infos.append(Chord_Match[0])
    chord_sequence.append(Chord_Match[2])
    feature_seq.append(next_skill+1)# H generation을 4로 했다.
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
        np.arange(13),
        p=prob[0]
    )
  return infos,chord_sequence