# 최대한 많고 깔끔한 조건문을 사용하여 Skill들을 정의해볼 것. Multilabel Classification의 가능성이 있다.
"""
Skill들의 음악학적인 특성 & 계산적인 특성을 적는 곳
'resting' : 포함하는 음이 0 또는 1개인 경우 resting으로 정의. 다른 Skill들은 겹칠 수 있으나 이 skill이 Label될 경우 그냥 resting 고정이다.
즉, Skilling Labeling은 'resting'이 아닌 경우에 진행된다.(삭제)
'repeating' : 전체 음 중 n% 이상 또는 n개를 제외한 경우가 전부 같은 음일 경우 repeating으로 정의
'up_steping' : 전체 음 중 n% 이상 또는 n개를 제외한 경우가 steping up 또는 같은 음, 즉 반음기준 3Note 이하로 상승하는 형태일 경우 up_steping으로 정의
'down_steping' : 전체 음 중 n% 이상 또는 n개를 제외한 경우가 steping down 또는 같은 음, 즉 반음기준 3Note 이하로 하강하는 형태일 경우 down_steping으로 정의
'up_leaping' : 전체 음 중 n% 이상 또는 n개를 제외한 경우가 leaping up, 즉 반음기준 3Note 이상으로 상승하는 형태일 경우 up_leaping으로 정의
'down_leaping' : 전체 음 중 n% 이상 또는 n개를 제외한 경우가 leaping down, 즉 반음기준 3Note 이상으로 하강하는 형태일 경우 down_leaping으로 정의
3Note에서 겹치는게 맞다. Multilabel Classification을 고안 중이기 때문.
'steping_twisting' : 음이 4개 이상이고, n개를 제외한 경우가 2Note 이하의 상승과 하강을 반복하는 형태일 경우 steping_twisting으로 정의
'leaping_twisting' : 음이 4개 이상이고, n개를 제외한 경우가 3Note 이상의 상승과 하강을 반복하는 형태일 경우 leaping_twisting으로 정의
'fast_rhythm' : 1 bar 내에 음이 9개 이상인 경우 fast_rhythm으로 정의.
'One_rhythm' :  모든 음이 지닌 연주의 real_time, 즉 음이 울리고 다음 음이 나올때 까지의 시간이 같으면 One_rhythm으로 정의
'triplet' : real_time기반해서 triplet이 존재하면(Note 3개) triplet으로 정의
'Staccato' : real_Duration_Time 기반해서 n% 이상의 음의 Duration이 0.16667(최소단위*4임)보다 작으면(매우 짧으면) Staccato로 정의
'continuing_rhythm' : pitch_change_list에서 'Rest'의 비율이 25퍼센트 이하면 continuing_rhythm으로 정의
첫 음 제외 실 음의 75%를 기준으로 잡는다.
5개 이상의 음이 있다면 1개를 제외하고 전부 조건에 맞아야하고,
9개 이상의 음이 있다면 2개를 제외하고 전부 조건에 맞아야하고...
4개 이하는 다 맞아야 한다.
ex) CDEF -> up_steping, CDED -> None, CDEFD-> up_steping.
다만 Leaping에 대해서는 많이 후해질 것 같다. 거의 50%가까이..?
"""

"""
예시로는..
contour[0] = ['Starting_Point', 'Rest', '4.0', '1.0', 'Rest', '2.0', 'Rest', '-7.0', 'Rest'] Note pitch의 변화를 쉼표를 포함하여 의미한다.
contour[1] = [0.83333, 0.16666999999999987, 0.5, 0.33333, 0.16666999999999987, 0.8333299999999999, 0.16666999999999987, 0.75, 0.25] Duration을 쉼표를 포함하여 의미한다.
contour[2] = ['Starting_Point', '4.0', '1.0', '2.0', '-7.0'] note pitch의 변화를 의미한다.
contour[3] = [1.0, 0.5, 0.5, 1.0, 1.0] 한 음의 실 연주시간을 의미한다.(다음 음이 나올때 까지의 시간)
contour[4] = [0.83333, 0.5, 0.33333, 0.8333299999999999, 0.75] Duration을 의미한다.
"""


def is_repeating(contour_list, exception_range):
    boolean_repeating = 0
    non_repeat = 0
    for elements in contour_list:
        if (elements is not 'Starting_Point'):
            if (elements != '0.0'):
                non_repeat += 1
    if (non_repeat <= exception_range):
        boolean_repeating = 1
    return boolean_repeating


def is_up_steping(contour_list, exception_range):
    balancing_param = 1
    boolean_up_steping = 0
    non_step_up = 0
    now_step_up = 0
    for elements in contour_list:
        if (elements is not 'Starting_Point'):
            if (now_step_up == 0):
                if (float(elements) < 0.5 or float(elements) > 4.5):
                    now_step_up = 0
                    non_step_up += 1
                else:
                    now_step_up = 1
            else:
                if (float(elements) < -0.5 or float(elements) > 4.5):
                    now_step_up = 0
                    non_step_up += 1
                else:
                    now_step_up = 1
    if (non_step_up <= exception_range + balancing_param):
        boolean_up_steping = 1
    return boolean_up_steping


def is_down_steping(contour_list, exception_range):
    boolean_down_steping = 0
    balancing_param = 0
    non_step_down = 0
    now_step_down = 0
    for elements in contour_list:
        if (elements is not 'Starting_Point'):
            if (now_step_down == 0):
                if (float(elements) > -0.5 or float(elements) < -4.5):
                    now_step_down = 0
                    non_step_down += 1
                else:
                    now_step_down = 1
            else:
                if (float(elements) > 0.5 or float(elements) < -4.5):
                    now_step_down = 0
                    non_step_down += 1
                else:
                    now_step_down = 1
    if (non_step_down <= exception_range + balancing_param):
        boolean_down_steping = 1
    return boolean_down_steping


def is_up_leaping(contour_list, exception_range):
    boolean_up_leaping = 0
    non_leap_up = 0
    for elements in contour_list:
        if (elements is not 'Starting_Point'):
            if (float(elements) < 3.5):
                non_leap_up += 1
    if (non_leap_up <= exception_range + 1):
        boolean_up_leaping = 1
    return boolean_up_leaping


def is_down_leaping(contour_list, exception_range):
    boolean_down_leaping = 0
    non_leap_down = 0
    for elements in contour_list:
        if (elements is not 'Starting_Point'):
            if (float(elements) > -3.5):
                non_leap_down += 1
    if (non_leap_down <= exception_range + 1):
        boolean_down_leaping = 1
    return boolean_down_leaping


def is_leaping_twisting(contour_list, exception_range):
    boolean_leaping_twisting = 0
    non_leap_twist = 0
    balancing_param = 1
    now_dir = 0  # 1for up, -1 for down
    for elements in contour_list:
        if (elements is not 'Starting_Point'):
            if (now_dir == 0):
                if (3.5 < float(elements)):
                    now_dir = 1
                elif (float(elements) < -3.5):
                    now_dir = -1
                else:
                    non_leap_twist += 1
            elif (now_dir == 1):
                if (float(elements) < -3.5):
                    now_dir = -1
                else:
                    now_dir = 0
                    non_leap_twist += 1
            elif (now_dir == -1):
                if (3.5 < float(elements)):
                    now_dir = 1
                else:
                    now_dir = 0
                    non_leap_twist += 1
    if (non_leap_twist <= exception_range + balancing_param):
        boolean_leaping_twisting = 1
    return boolean_leaping_twisting


def is_steping_twisting(contour_list, exception_range):
    boolean_steping_twisting = 0
    non_step_twist = 0
    now_dir = 0  # 1for up, -1 for down
    for elements in contour_list:
        if (elements is not 'Starting_Point'):
            if (now_dir == 0):
                if (0 < float(elements) and float(elements) < 2.5):
                    now_dir = 1
                elif (-2.5 < float(elements) and float(elements) < 0):
                    now_dir = -1
                else:
                    non_step_twist += 1
            elif (now_dir == 1):
                if (-2.5 < float(elements) and float(elements) < 0):
                    now_dir = -1
                else:
                    now_dir = 0
                    non_step_twist += 1
            elif (now_dir == -1):
                if (0 < float(elements) and float(elements) < 2.5):
                    now_dir = 1
                else:
                    now_dir = 0
                    non_step_twist += 1
    if (non_step_twist <= exception_range):
        boolean_steping_twisting = 1
    return boolean_steping_twisting


def is_one_rhythm(contour_list, exception_range):
    boolean_one_rhythm = 0
    non_same_rhythm = 0

    first_rhythm = contour_list[0]
    for rhythms in contour_list:
        if (rhythms != first_rhythm):
            non_same_rhythm = 1
    boolean_one_rhythm = 1 - non_same_rhythm
    return boolean_one_rhythm


def is_triplet(contour_list, exception_range):
    boolean_triplet = 0
    now_triplet = 0
    for rhythms in contour_list:
        rhythms = float(rhythms)
        if (rhythms % 0.015625 > 0.001):
            if (now_triplet == 1):
                boolean_triplet = 1
            now_triplet += 1
        else:
            now_triplet = 0

    return boolean_triplet


def is_staccato(contour_list, exception_range):
    boolean_staccato = 0
    ranges = len(contour_list) // 2
    staccato_num = 0
    for times in contour_list:
        if (times < 0.2):
            staccato_num += 1
    if (staccato_num >= ranges):
        boolean_staccato = 1
    return boolean_staccato


def is_continuing_rhythm(contour_list):
    boolean_continuing_rhythm = 0
    length = len(contour_list)
    rest_num = 0
    for elements in contour_list:
        if (elements == 'Rest'):
            rest_num += 1
    if (rest_num <= 0.5):
        boolean_continuing_rhythm = 1
    return boolean_continuing_rhythm


def contour_to_label(contour):
    labels = []
    totnum = len(contour[2])  # 실 음의 갯수이다.
    exception_range = (totnum - 1) // 4
    exception_range2 = (totnum - 1) // 3
    if (len(contour[2]) < 2.5):
        pass  # 원래는 resting이라는 Label을 append했으나 Control이 까다롭다
    else:
        if (is_repeating(contour[2], exception_range2)):
            labels.append('repeating')

        if (is_up_steping(contour[2], exception_range)):
            if (len(contour[2]) > 3):
                labels.append('up_steping')

        if (is_down_steping(contour[2], exception_range)):
            if (len(contour[2]) > 3):
                labels.append('down_steping')

        if (is_up_leaping(contour[2], exception_range2)):
            labels.append('up_leaping')

        if (is_down_leaping(contour[2], exception_range2)):
            labels.append('down_leaping')

        if (is_steping_twisting(contour[2], exception_range2)):
            if (len(contour[2]) > 3):
                labels.append('steping_twisting')

        if (is_leaping_twisting(contour[2], exception_range2)):
            if (len(contour[2]) > 3):
                labels.append('leaping_twisting')

        if (len(contour[2]) > 8.5):
            labels.append('fast_rhythm')

        if (is_one_rhythm(contour[3], exception_range)):
            labels.append('One_rhythm')

        if (is_triplet(contour[3], exception_range2)):
            labels.append('triplet')

        if (is_staccato(contour[4], exception_range2)):
            labels.append('staccato')

        if (is_continuing_rhythm(contour[0])):
            labels.append('continuing_rhythm')

    if (len(labels) == 0):
        labels.append('no skills')  # classifier과 cGan에서 사용은 하되, 추후 음원 제작에서 사용을 안한다.

    return labels

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
      real_time_list.append(lists[0]-now_rhythm)
    now_rhythm=lists[0]
    a+=1
    if (first_time*1.001<lists[0]):#smoothing for case like first time=5.00001, lists[0]=5.0000..
      resting_time=lists[0]-first_time
      duration_list.append(resting_time)
      first_time=lists[0]
      pitch_change_list.append('Rest')
    if (now_pitch==1000):
      pitch_change_list.append('Starting_Point')
      real_pitch_list.append('Starting_Point')
      real_duration_list.append(lists[3])
      duration_list.append(lists[3])
      first_time=first_time+lists[3]
      now_pitch=lists[1]
      a+=1
    else:
      pitch_change=lists[1]-now_pitch
      pitch_change_list.append(str(pitch_change))#나중에 int로 바꿔쓸 것. 자료형 터지는거 때문에 우선 스트링.
      duration_list.append(lists[3])
      real_duration_list.append(lists[3])
      first_time=first_time+lists[3]
      now_pitch=lists[1]
      real_pitch_list.append(str(pitch_change))
  if (first_time*1.001<starting_number+one_bar_number*(j+1)):
    pitch_change_list.append('Rest')#마지막 Rest
    duration_list.append(starting_number+one_bar_number*(j+1)-first_time)
  if(len(bar)!=0):
    real_time_list.append(starting_number+one_bar_number*(j+1)-now_rhythm)
  contour.append(pitch_change_list)
  contour.append(duration_list)
  contour.append(real_pitch_list)
  contour.append(real_time_list)
  contour.append(real_duration_list)
  #something
  return contour