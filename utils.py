import json
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import scipy as sp
import scipy.ndimage

def get_meta(filename):
  with open('PPDD-Sep2018_sym_mono_large/descriptor/'+filename) as json_file:
    meta_data = json.load(json_file)
  return meta_data
def csv_to_array(csvlist,jsonlist):
    bar_list=[]
    one_bar_number_list=[]
    starting_number_list=[]
    for i,csvs in enumerate(tqdm(csvlist)):
      a=np.array([list(map(float,csvs.columns))])
      b=np.array(csvs.values)
      csvarray=np.concatenate((a,b),axis=0)
      if('timeSignature' not in jsonlist[i]):
        jsonlist[i]['timeSignature']=[4,4]
      one_bar_number=jsonlist[i]['timeSignature'][0]
      bar_number=(csvarray[-1][0]-csvarray[0][0])//one_bar_number+1
      bar_info_list=[]
      for i in range(int(bar_number)):
        starting_bar_time=csvarray[0][0]+i*one_bar_number
        bar_info_list.append(csvarray[np.where( (starting_bar_time<=csvarray[:,0]) & (csvarray[:,0]<starting_bar_time+one_bar_number) )])
      bar_list.append(bar_info_list)
      one_bar_number_list.append(one_bar_number)
      starting_number_list.append(csvarray[0][0])
    return bar_list, one_bar_number_list, starting_number_list

def midi_to_array(prettymidilist,jsonlist):
    bar_list = []
    one_bar_number_list = []
    starting_number_list = []
    for i, songs in enumerate(prettymidilist):
        for instrument in songs.instruments:  # In this implementation, there is only one instruments.
            csvarray = []
            for note in instrument.notes:  # 3
                row = [note.start * 2, note.pitch, note.pitch, (note.end - note.start) * 2,
                       0]  # *2 for use minimum unit as one bar.
                csvarray.append(row)
        csvarray = np.array(csvarray)
        if ('timeSignature' not in jsonlist[i]):
            jsonlist[i]['timeSignature'] = [4, 4]
        if (jsonlist[i]['timeSignature'] != [4, 4]):  # Line 50~52 means that we use [4,4] Time signature only. If you want all data, then just delete 50~52 and set the indentation.
            pass
        else:
            one_bar_number = jsonlist[i]['timeSignature'][0]
            bar_number = (csvarray[-1][0] - csvarray[0][0]) // one_bar_number + 1
            bar_info_list = []
            for j in range(int(bar_number)):
                starting_bar_time = csvarray[0][0] + j * one_bar_number
                bar_info_list.append(csvarray[np.where(
                    (starting_bar_time <= csvarray[:, 0]) & (csvarray[:, 0] < starting_bar_time + one_bar_number))])
            bar_list.append(bar_info_list)
            one_bar_number_list.append(one_bar_number)
            starting_number_list.append(csvarray[0][0])
    return bar_list, one_bar_number_list, starting_number_list


def nearest_time(time, minimum_size):
    # 혹시나 값이 조금 벗어나는 엇박 음을 가까운 최소단위로 Shifting한다.
    # 다만 엇박 관련 Skill Detecting을 따로 고려할시 코드를 수정할 수 있다.
    num_to_multiply = time / minimum_size
    num_to_multiply = int(num_to_multiply)
    left_time = num_to_multiply * minimum_size
    right_time = left_time + minimum_size
    if (time - left_time >= right_time - time):
        return right_time
    return left_time


def bar_to_matrix3(bar,one_bar_number,starting_number,i,minimum_time):

  #For time signature 8/6 one_bar_number is 8.
  #There waere bar_tp_matrix 1,2 function, but I don't use them..

  init=np.zeros((24,minimum_time))
  minimum_size=one_bar_number/minimum_time
  zero_time=starting_number+one_bar_number*i
  min_height=500
  for lists in bar:
    if min_height>lists[1]:
      min_height=lists[1]
  for j,lists in enumerate(bar):
    #lists[0]is time, lists[1] is Note pitch, lists[3]is Duration.
    point=int((nearest_time(lists[0],minimum_size)-zero_time)/minimum_size)
    length=int(round(lists[3]/minimum_size))
    if (length>3):
      length=length-1#Handle for notes with repeating rhythm. like 3-3 two notes and 6 one notes.
    if (point+length>minimum_time-1):
      length=minimum_time-1-point # if One note is playing
    height=lists[1]-min_height
    """
    while(height>23):
      height=height-12 
    """
    if (height<24):#two choices for handling notes that has pitch difference above 23, delete or shifting.
      init[23-int(height)][point:point+length]+=1
  return init

def plot_bar(bar_matrix_list2):
    H = bar_matrix_list2[0][0]

    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(H)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()


def set_labels():
    labels = []
    skills_pitch = ['repeating', 'up_steping', 'down_steping', 'up_leaping', 'down_leaping', 'steping_twisting',
                    'leaping_twisting', 'dummy']
    skills_timing = ['fast_rhythm', 'dummy']
    skills_triplet = ['triplet', 'dummy']
    skills_one_rhythm = ['One_rhythm', 'dummy']
    skills_staccato = ['staccato', 'continuing_rhythm', 'dummy']
    for pitch in skills_pitch:
        for timing in skills_timing:
            for triplet in skills_triplet:
                for one_rhythm in skills_one_rhythm:
                    for staccato in skills_staccato:
                        label_tuple = []
                        if pitch is not 'dummy':
                            label_tuple.append(pitch)
                        if timing is not 'dummy':
                            label_tuple.append(timing)
                        if triplet is not 'dummy':
                            label_tuple.append(triplet)
                        if one_rhythm is not 'dummy':
                            label_tuple.append(one_rhythm)
                        if staccato is not 'dummy':
                            label_tuple.append(staccato)
                        if len(label_tuple) == 0:
                            label_tuple.append(
                                'no skills')  # no skills label is used for training classifier and generator, but not used for real generation.
                        label_tuple = tuple(label_tuple)

                        labels.append(label_tuple)
    return labels
def get_best_results(testresult,test_label2):  #for optimizing.
  mlb = MultiLabelBinarizer()
  labels = set_labels()
  mlb.fit(labels)
  classnum={}
  testnum={}
  resultmat=[]
  bestmat=[]
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
  sigma_y = 1.0
  sigma_x = 1.0
  inputmat=matrix
  # Apply gaussian filter
  sigma = [sigma_y, sigma_x]
  y = sp.ndimage.filters.gaussian_filter(inputmat, sigma, mode='constant')
  return y
