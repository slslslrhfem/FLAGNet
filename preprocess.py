import os
from util import *
from tqdm import tqdm
import mido
import pretty_midi
import music21
import numpy as np
import copy
from musical_skill_definition import contour_to_label
from hyperparameter import Hyperparams as hp
import gc

def get_midilist(dir):
  pop909namelist=os.listdir(dir)
  pop909namelist.sort()
  midilist=[]
  prettymidilist=[]
  print("pretty_midi processing...")
  for names in tqdm(pop909namelist): # May get killed with whole list..
    try:
      midi_path=dir+"/"+str(names)+"/"+str(names)+".mid"
      mid = mido.MidiFile(midi_path, clip=True)
      midilist.append(mid)
      prettymid=pretty_midi.PrettyMIDI(midi_path)
      prettymidilist.append(prettymid)
    except:
      pass
  return prettymidilist, midilist

def get_music21list(dir):
  pop909namelist=os.listdir(dir)
  pop909namelist.sort()
  music21list=[]
  print("music21_midi processing...")
  for names in tqdm(pop909namelist): # May get killed with whole list..
    try:
      mf = music21.midi.MidiFile()
      mf.open(dir+"/"+str(names)+"/"+str(names)+".mid")
      mf.read()
      mf.close()
      music21list.append(music21.midi.translate.midiFileToStream(mf))
    except:
      pass
  return music21list


def get_bar_lists(music21list, prettymidilist, meta_data):
    bar_list=[]
    one_bar_number_list=[]
    starting_number_list=[]
    for i,songs in enumerate(music21list):
        TS = get_ts(i,meta_data)
        if(TS!=[2,2]):#This code is used for using only 4/4 time signature.
            # This processsing method can handle whole time signature, but I highly recommend to use only one type of TS. Data with 4/4 and 3/4 has different relative time difference.
            # Note that Triplet skill label fits for 4/4.
            pass
        else:
            TS=double(TS)
            pretty_song=prettymidilist[i]
            totcsvarray=[]
            for j,instrument in enumerate(pretty_song.instruments): #2
                if j==0:#melody만
                    issame=1
                    top = music21list[i].parts[j].flat.notes 
                    x, y, parent_element, duration, velocity = extract_notes(top) # use y as pitch
                    for k, element in enumerate(x):
                        x[k]=float(element)
                    csvarray=[]
                    for j in range(len(x)): #3
                        row=[y[j], x[j], duration[j], velocity[j]]
                        csvarray.append(row)
            one_bar_number=TS[0]
            bar_number=(csvarray[-1][1]-csvarray[0][1])//one_bar_number+1
            bar_info_list=[]
            
            started=0
            csvarray=np.array(csvarray)
            for i in range(int(bar_number)):
                starting_bar_time=i*one_bar_number#+csvarray[0][0] -> With this addition, then First note always the First timing of first bar. but POP909 has well-organized bar data, So I didn't take any risks.
                final_array=csvarray[np.where( (starting_bar_time<=csvarray[:,1]) & (csvarray[:,1]<starting_bar_time+one_bar_number) )]
                if (len(final_array) !=0 or started==1):
                    if started==0:
                        starting_number_list.append(starting_bar_time)
                        started=1
                    bar_info_list.append(final_array)
                
            bar_list.append(bar_info_list)
            one_bar_number_list.append(one_bar_number)
    return bar_list, one_bar_number_list, starting_number_list

def bar_list_to_primining(bar_list,starting_number_list,one_bar_number_list):
  primining_matrix_list=copy.deepcopy(bar_list)
  for i,songs in enumerate(primining_matrix_list):
    for j,bar in enumerate(songs):
      if len(bar_list[i][j]) != 0:
        primining_pitch=[]
        primining_time=[]
        primining_matrix=[]
        if j==0:
          first_pitch = bar_list[i][j][0][0]
        else:
          if len(bar_list[i][j-1])>0:
            first_pitch = bar_list[i][j-1][-1][0]
          else:
            first_pitch = bar_list[i][j][0][0]
        first_time = starting_number_list[i]+one_bar_number_list[i]*j
        primining_bar_list=[]
        for t in range(8):
          if j+t-8>=0:
            primining_bar_list.append(j+t-8)
        for k in primining_bar_list:
          for notes in bar_list[i][k]:
              primining_pitch.append(notes[0]-first_pitch)
              primining_time.append(notes[1]-first_time)
      else:
        primining_pitch=[]
        primining_time=[]
        primining_matrix=[]
      primining_matrix.append(np.array(primining_pitch))
      primining_matrix.append(np.array(primining_time))
      primining_matrix_list[i][j]=primining_matrix
  return primining_matrix_list

def bar_to_matrix(bar,one_bar_number,starting_number,i,minimum_time):
  init=np.zeros((24,minimum_time))
  minimum_size=one_bar_number/minimum_time
  zero_time=starting_number+one_bar_number*i
  min_height=500
  for lists in bar:
    if min_height>lists[0]:
      min_height=lists[0]
  for i,lists in enumerate(bar):    
      point=int((nearest_time(lists[1],minimum_size)-zero_time)/minimum_size)
      length=int(round(2*lists[2]/minimum_size))
      if length==0:
        length=1
      if (length>3 and point+length != minimum_time-1):
        length=length-1#For Handling repeating notes.
      if (point+length>minimum_time-1):
        length=minimum_time-point
      height=lists[0]-min_height
      """
      while(height>23):
        height=height-12 
      """
      if (height<24):#two choices for handling notes that has pitch difference above 23, delete or shifting. 
      #If you want shift them then uncomment above 2 line of codes. 
        init[23-int(height)][point:point+length]+=lists[3]
        init=np.array(init)
        init=np.where(init>128,128,init) # For Hanling some Overlapping Notes.
  return init

def bar_to_contour(bar,one_bar_number,starting_number,j):
  contour=[]
  pitch_change_list=[]
  duration_list=[]
  real_pitch_list=[]
  real_time_list=[]
  real_duration_list=[]
  now_pitch=1000
  first_time=starting_number+one_bar_number*j
  a=0
  for lists in bar:
    if(a!=0): 
      real_time_list.append(lists[1]-now_rhythm)
    now_rhythm=lists[1]
    a+=1
    if (first_time*1.001<lists[1]):#smoothing for case like first time=5.00001, lists[0]=5.0000..
      resting_time=lists[1]-first_time
      duration_list.append(resting_time)
      first_time=lists[1]
      pitch_change_list.append('Rest')
    if (now_pitch==1000):
      pitch_change_list.append('Starting_Point')
      real_pitch_list.append('Starting_Point')
      real_duration_list.append(lists[2])
      duration_list.append(lists[2])
      first_time=first_time+lists[2]
      now_pitch=lists[0]
      a+=1
    else:
      pitch_change=lists[0]-now_pitch
      pitch_change_list.append(str(pitch_change))
      duration_list.append(lists[2])
      real_duration_list.append(lists[2])
      first_time=first_time+lists[2]
      now_pitch=lists[0]
      real_pitch_list.append(str(pitch_change))
  if (first_time*1.001<starting_number+one_bar_number*(j+1)):
    pitch_change_list.append('Rest')
    duration_list.append(starting_number+one_bar_number*(j+1)-first_time)
  if(len(bar)!=0):
    real_time_list.append(starting_number+one_bar_number*(j+1)-now_rhythm)
  contour.append(pitch_change_list)
  contour.append(duration_list)
  contour.append(real_pitch_list)
  contour.append(real_time_list)
  contour.append(real_duration_list)
  return contour

def preprocessing():
    music21list=get_music21list("POP909")

    prettymidilist, midilist = get_midilist("POP909")

    meta_data = get_meta()
    #above 3 method simply get datas for training.

    bar_list, one_bar_number_list, starting_number_list = get_bar_lists(music21list, prettymidilist, meta_data)
    #this method get bar-by-bar midi information list. one_bar_number_list get time-signature data for songs. ( 4 for 4/4, 3 for 3/4.. etc )

    primining_matrix_list = bar_list_to_primining(bar_list, starting_number_list, one_bar_number_list)
    #this method get primining relative bar midi information.

    bar_matrix_list3=copy.deepcopy(bar_list)
    for i,songs in enumerate(bar_matrix_list3):
        for j,bar in enumerate(songs):
            matrix3=bar_to_matrix(bar,one_bar_number_list[i],starting_number_list[i],j,hp.Minimum_time)
            bar_matrix_list3[i][j]=matrix3

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
    for i,songs in enumerate(bar_list):
        for j,bar in enumerate(songs):
            contour=bar_to_contour(bar,one_bar_number_list[i],starting_number_list[i],j)
            bar_contour_list[i][j]=contour

    bar_label_list=copy.deepcopy(bar_contour_list)
    for i,songs in enumerate(bar_contour_list):
        for j,contour in enumerate(songs):
            label=contour_to_label(contour) 
            bar_label_list[i][j]=label

    all_matrix=[]
    all_labels=[]
    all_updown_labels=[]
    all_primining_notes=[]
    for songs in bar_label_list:
        for label in songs:
            label=np.array(label)
            all_labels.append(label)
    for songs in bar_matrix_list3:
        for matrix in songs:
            matrix=matrix.reshape(24,hp.Minimum_time,1)
            all_matrix.append(matrix)
    for songs in bar_updown_list:
        for label in songs:
            all_updown_labels.append(label)
    for songs in primining_matrix_list:
        for label in songs:
            all_primining_notes.append(label)
            
    np.save('preprocessing/bar_matrix_lists',bar_matrix_list3)
    np.save('preprocessing/all_matrix',all_matrix)
    np.save('preprocessing/all_labels',all_labels)
    np.save('preprocessing/all_updown_labels',all_updown_labels)
    np.save('preprocessing/all_primining_notes',all_primining_notes)
