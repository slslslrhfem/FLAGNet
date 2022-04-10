from hyperparameter import Hyperparams as hp
"""
Information of skill labels

The following contents are labels and explanations. n is a hyper-parameter that we can
control to make labels balanced. In implementation, we used 66% for rare case like ’up leaping’,
’down leaping’. and others use 75%. Some case like ’twisting’ has some different condition, but same as others, n is hyper parameter.


’repeating’: defined as repeating when all but n% or more of the total notes are the same.
’up stepping’: defined as up stepping if more than n% of the total notes, or excluding n,
are stepping up or the same note, i.e. rising below 3Note on a semi-negative basis.
’down stepping’: defined as down stepping if more than n% of the total notes, or excluding n, are stepping down or the same sound, i.e. falling below 3Note on a semi-negative basis.
’up leaping’: defined as up leaving if more than n% of the total notes, or excluding n,
are leaping up, i.e. if they rise above 3Note on a semi-negative basis.
’down leaping’: defined as down leaving if more than n% of the total notes, or excluding
n, are leaping down, i.e. lower than 3Note on a semi-negative basis.
’stepping twisting’: defined as stepping twisting if there are more than four notes, excluding n, in the form of repeated rises and falls of less than 2Note.
’leaping twisting’: defined as leaping twisting if there are more than four notes, excluding n, in the form of repeating the rise and fall of more than 3Note.
’fast rhythm’: defined as fast rhythm if there are more than 9 notes within 1 bar.
’One rhythm’: The real-time of all notes, that is, if the time until the next note is the
same, define it as One rhythm.
’triplet’: defined as a triplet if a triplet exists based on real-time. For a generation with a
minimum unit as 16th notes, this skill is ignored.
’Staccato’: Based on Duration Time, if the negative Duration of n% or more is less than
0.16667 (minimum unit for this study.), defined as Staccato.
’continuning rhythm’: If the ratio of ’Rest’ in the pitch change list is less than 25%, it is
defined as continuing rhythm.
’no skills’: if the bar has no property among these skills, then we label this bar as
’no skills’ and do not use for generation.
While constructing multi-label sets, we clear the impossible cases like case which have
both ’up stepping’ and ’down stepping’, to improve the performance of the classifier. Also, this labeling method occurs noise for some cases, so we considered this classifier as a noisy label classifier.
and try to denoise.


Example and Contents of contour list:

contour[0] = ['Starting_Point', 'Rest', '4.0', '1.0', 'Rest', '2.0', 'Rest', '-7.0', 'Rest'] Change of Note pitch, Including Rest
contour[1] = [0.83333, 0.16666999999999987, 0.5, 0.33333, 0.16666999999999987, 0.8333299999999999, 0.16666999999999987, 0.75, 0.25] Duration, Including Rest
contour[2] = ['Starting_Point', '4.0', '1.0', '2.0', '-7.0'] note pitch of each note
contour[3] = [1.0, 0.5, 0.5, 1.0, 1.0] Real Play time for each note(Time until next notes appear)
contour[4] = [0.83333, 0.5, 0.33333, 0.8333299999999999, 0.75] Duration of each note

if above contour's first note is C4, then it has notes 
(C4, duration : 0.83333, time : 0), (E4, duration : 0.5, time : 1) (F4, duration : 0.33333, time : 1.5), (G4, duration : 0.83333, time : 2.0), (C4, duration : 0.75, time : 3.0)
"""

def is_repeating(contour_list,exception_range):
  boolean_repeating=0
  non_repeat=0
  for elements in contour_list:
    if (elements is not 'Starting_Point'):
      if(elements != '0.0'):
        non_repeat+=1
  if (non_repeat<=exception_range):
    boolean_repeating=1
  return boolean_repeating

def is_up_steping(contour_list,exception_range):
  balancing_param=1
  boolean_up_steping=0
  non_step_up=0
  now_step_up=0
  for elements in contour_list:
    if (elements is not 'Starting_Point'):
      if (now_step_up==0):
        if (float(elements)<0.5 or float(elements)>4.5):
          now_step_up=0
          non_step_up+=1
        else:
          now_step_up=1
      else:
        if(float(elements)<-0.5 or float(elements)>4.5):
          now_step_up=0
          non_step_up+=1
        else:
          now_step_up=1
  if(non_step_up<=exception_range+balancing_param):
    boolean_up_steping=1
  return boolean_up_steping

def is_down_steping(contour_list,exception_range):
  boolean_down_steping=0
  balancing_param=0
  non_step_down=0
  now_step_down=0
  for elements in contour_list:
    if (elements is not 'Starting_Point'):
      if (now_step_down==0):
        if (float(elements)>-0.5 or float(elements)<-4.5):
          now_step_down=0
          non_step_down+=1
        else:
          now_step_down=1
      else:
        if(float(elements)>0.5 or float(elements)<-4.5):
          now_step_down=0
          non_step_down+=1
        else:
          now_step_down=1
  if(non_step_down<=exception_range+balancing_param):
    boolean_down_steping=1
  return boolean_down_steping

def is_up_leaping(contour_list,exception_range):
  boolean_up_leaping=0
  non_leap_up=0
  for elements in contour_list:
    if (elements is not 'Starting_Point'):
      if (float(elements)<3.5):
        non_leap_up+=1
  if (non_leap_up<=exception_range+1):
    boolean_up_leaping=1
  return boolean_up_leaping

def is_down_leaping(contour_list,exception_range):
  boolean_down_leaping=0
  non_leap_down=0
  for elements in contour_list:
    if (elements is not 'Starting_Point'):
      if (float(elements)>-3.5):
        non_leap_down+=1
  if (non_leap_down<=exception_range+1):
    boolean_down_leaping=1
  return boolean_down_leaping

def is_leaping_twisting(contour_list,exception_range):
  boolean_leaping_twisting=0
  non_leap_twist=0
  balancing_param=1
  now_dir=0 #1for up, -1 for down
  for elements in contour_list:
    if (elements is not 'Starting_Point'):
      if (now_dir==0):
        if (3.5<float(elements)):
          now_dir=1
        elif (float(elements) <-3.5):
          now_dir=-1
        else:
          non_leap_twist+=1
      elif (now_dir==1):
        if (float(elements) <-3.5):
          now_dir=-1
        else:
          now_dir=0
          non_leap_twist+=1
      elif (now_dir==-1):
        if (3.5<float(elements)):
          now_dir=1
        else:
          now_dir=0
          non_leap_twist+=1
  if(non_leap_twist<=exception_range+balancing_param):
    boolean_leaping_twisting=1
  return boolean_leaping_twisting

def is_steping_twisting(contour_list,exception_range):
  boolean_steping_twisting=0
  non_step_twist=0
  now_dir=0 #1for up, -1 for down
  for elements in contour_list:
    if (elements is not 'Starting_Point'):
      if (now_dir==0):
        if (0<float(elements) and float(elements) <2.5):
          now_dir=1
        elif (-2.5<float(elements) and float(elements) <0):
          now_dir=-1
        else:
          non_step_twist+=1
      elif (now_dir==1):
        if (-2.5<float(elements) and float(elements) <0):
          now_dir=-1
        else:
          now_dir=0
          non_step_twist+=1
      elif (now_dir==-1):
        if (0<float(elements) and float(elements) <2.5):
          now_dir=1
        else:
          now_dir=0
          non_step_twist+=1
  if(non_step_twist<=exception_range):
    boolean_steping_twisting=1
  return boolean_steping_twisting

def is_one_rhythm(contour_list,exception_range):
  boolean_one_rhythm=0
  non_same_rhythm=0

  first_rhythm=contour_list[0]
  for rhythms in contour_list:
    if (rhythms != first_rhythm):
      non_same_rhythm=1
  boolean_one_rhythm=1-non_same_rhythm
  return boolean_one_rhythm

def is_triplet(contour_list,exception_range):
  boolean_triplet=0
  now_triplet=0
  for rhythms in contour_list:
    rhythms=float(rhythms)
    if (rhythms%0.015625>0.001):
      if(now_triplet==1):
        boolean_triplet=1
      now_triplet+=1
    else:
      now_triplet=0
      
  return boolean_triplet

def is_staccato(contour_list,exception_range):
  boolean_staccato=0
  ranges=len(contour_list)//2
  staccato_num=0
  for times in contour_list:
    if (times<0.26): #if Minimum time = 16 -> minimum length == 0.25
      staccato_num+=1
  if (staccato_num>=ranges):
    boolean_staccato=1
  return boolean_staccato

def is_continuing_rhythm(contour_list):
  boolean_continuing_rhythm=0
  length=len(contour_list)
  rest_num=0
  for elements in contour_list:
    if (elements=='Rest'):
      rest_num+=1
  if (rest_num<=0.5):
    boolean_continuing_rhythm=1
  return boolean_continuing_rhythm

def contour_to_label(contour):
  labels=[]
  totnum=len(contour[2]) #실 음의 갯수이다.
  exception_range=(totnum-1)//3
  exception_range2=(totnum-1)//2
  if (len(contour[2])<2.5):
    pass#원래는 resting이라는 Label을 append했으나 Control이 까다롭다
  else:
    if (is_repeating(contour[2],exception_range2)):
      labels.append('repeating')

    if (is_up_steping(contour[2],exception_range)):
      if (len(contour[2])>3):
        labels.append('up_steping')

    if (is_down_steping(contour[2],exception_range)):
      if (len(contour[2])>3):
        labels.append('down_steping')

    if (is_up_leaping(contour[2],exception_range2)):
      labels.append('up_leaping')
    
    if (is_down_leaping(contour[2],exception_range2)):
      labels.append('down_leaping')

    if (is_steping_twisting(contour[2],exception_range2)):
      if (len(contour[2])>3):
        #labels.append('steping_twisting')
        pass
    if (is_leaping_twisting(contour[2],exception_range2)):
      if (len(contour[2])>3):
        #labels.append('leaping_twisting')
        pass
    if (len(contour[2])>8.5):
      labels.append('fast_rhythm')

    if (is_one_rhythm(contour[3],exception_range2)):
      #labels.append('One_rhythm')
      pass
    if (is_triplet(contour[3],exception_range2) and hp.Minimum_time//3 == 0):
      labels.append('triplet')

    if (is_staccato(contour[4],exception_range2)):
      #labels.append('staccato')  
      pass
    if (is_continuing_rhythm(contour[0])):
      #labels.append('continuing_rhythm')  
      pass
  if (len(labels)==0):
    labels.append('no skills') # Don't use this label at generation. 


  return labels