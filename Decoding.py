import numpy as np
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.pyplot as pp
C_chord=[1,0,1,0,1,1,0,1,0,1,0,1]
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

def left_shifting(matrix,detected_peak):
  #input은 detected_peak가 들어와야한다.
  dots=[]
  for j in range(len(detected_peak)):
    pos=[]
    for i in range(len(detected_peak[0])):#for 문의 순서를 이렇게 지정해야 Handle이 가능하다.
      if(detected_peak[i][j]==1):
        if(j%2!=0 and j%3!=0):
          j=j-1
        if(len(pos)==0):
          pos=[i,j]
        else:
          if (matrix[pos[0]][pos[1]]<matrix[i][j]):
            pos=[i,j]
    if(len(pos)!=0):
      dots.append(pos)
  dots_with_length=[]
  starting_points=[]
  durations=[]
  velocities=[]
  for position in dots:
    velocities.append(matrix[position[0]][position[1]])
    length_val=0
    while True:
      if(length_val==0):
        starting_points.append([position[0],position[1]])
      if(position[1]+length_val>23):
        break
      elif(matrix[position[0]][position[1]+length_val]!=0):
        dots_with_length.append([position[0],position[1]+length_val])
      else:
        break
      length_val+=1
    durations.append(length_val)
  result=np.zeros_like(matrix)
  for position in dots_with_length:
    result[position[0]][position[1]]=1
  return result,np.array(starting_points),np.array(durations),np.array(velocities)

def chord_matching(shifted_matrix,chord,last_pitch,direction):
  #input은 left_shifting의 output을 그대로 넣고, chord(C,C#,D,D#....,B), 마지막 리듬의 pitch를 숫자로, 그리고 진행 방향을 direction(1,0)으로 받는다.
  #output은 MIDI기반의 decoding이 되도록 정보 기반으로, 그리고 last_pitch까지 넣어준다.
  #또한 출력 그림과 상관없이, 그냥 Note Pitch를 그냥 Matrix에 넣어준다.
  before_matrix=shifted_matrix[0]
  if (last_pitch<40):
    direction=1
  if (last_pitch>80):
    direction=0
  if chord not in chords:
    now_chord='C'
  now_chord=chord
  chord_diff=chords.index(now_chord)
  now_chord_list=np.roll(C_chord,chord_diff)#[1,0,1,0,1...] 이런 set인데 여기에 맞추면 함수에 넣은 chord에 맞게 된다.
  #나중에 그냥 chord뿐만 아니라 화음의 종류까지 고민하게 된다면 이걸 잘 쓰면 된다.
  pitch_set=[]
  for pitchs in shifted_matrix[1]:
    pitch_set.append(23-pitchs[0])#위 그림 기준 7,17,9,7,3,5 이렇게 들감
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
  for sets in possible_set:
    score=0
    for pitchs in sets:
      score+=now_chord_list[pitchs%12]
    possible_set_score.append(score)
  final_set=possible_set[np.argmax(possible_set_score)]
  return_val=[]
  for i,sets in enumerate(final_set):
    if(now_chord_list[sets%12]==1):
      final=[]
      final.append(sets)
      final.append(shifted_matrix[1][i][1])
      final.append(shifted_matrix[2][i])
      final.append(shifted_matrix[3][i])
      return_val.append(final)
  return return_val, sets
def generation_info(start_skill,length,chord):
  infos=[]
  H=generate_images(G,1,start_skill)
  shifted=left_shifting(matrix_cleaner(H),detect_peaks(matrix_cleaner(H)))
  infos.append(chord_matching(shifted,chord,48,1)[0]) # 48은 last_pitch정보, 1은 updown precition, 즉 direction 관련 정보다.
  last_pitch=chord_matching(shifted,chord,48,1)[1]

  feature_seq=[]
  feature_seq.append(start_skill+1)#RNN Predict Pad 전에 1을 더해야한다.
  feature_seq_pad=sequence.pad_sequences(np.array([feature_seq]),maxlen=10)
  prediction=RNNmodel.predict(feature_seq_pad)
  updown_prediction=updown_classifier.predict(shifted[0].reshape(1,24,24,1))
  if(updown_prediction[0][0]>=updown_prediction[0][1]):
    updown_prediction=1
  else:
    updown_prediction=0
  prob=prediction/prediction.sum()
  next_skill=np.random.choice(
      np.arange(12),
      p=prob[0]
  )
  for i in range(length-1):
    H=generate_images(G,1,next_skill)
    shifted=left_shifting(matrix_cleaner(H),detect_peaks(matrix_cleaner(H)))
    infos.append(chord_matching(shifted,chord,last_pitch,updown_prediction)[0])
    feature_seq.append(next_skill+1)# H generation을 4로 했다.
    feature_seq_pad=sequence.pad_sequences(np.array([feature_seq]),maxlen=10)
    prediction=RNNmodel.predict(feature_seq_pad)
    updown_prediction=updown_classifier.predict(shifted[0].reshape(1,24,24,1))
    if(updown_prediction[0][0]>=updown_prediction[0][1]):
      updown_prediction=1
    else:
      updown_prediction=0
    prob=prediction/prediction.sum()
    next_skill=np.random.choice(
        np.arange(12),
        p=prob[0]
    )
  return infos