import json
import tqdm
import numpy as np
import matplotlib.pyplot as plt
def get_meta(filename):
  with open('PPDD-Sep2018_sym_mono_small/PPDD-Sep2018_sym_mono_small/descriptor/'+filename) as json_file:
    meta_data = json.load(json_file)
  return meta_data
def csv_to_array(csvlist,jsonlist):
    bar_list=[]
    one_bar_number_list=[]
    starting_number_list=[]
    for i,csvs in enumerate(tqdm(csvlist)):
      a=np.array([list(map(float,csvs.columns))])#column에도 숫자가 들어가 있어서.. 경우에 따라 조절한다
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
    for i, songs in enumerate(prettymidilist[:2000]):  # 곡마다. Ram 관련 이슈로 2000으로 줄인다.
        for instrument in songs.instruments:  # 2. 어차피 instrument하나
            csvarray = []
            for note in instrument.notes:  # 3
                row = [note.start * 2, note.pitch, note.pitch, (note.end - note.start) * 2,
                       0]  # *2를 해줘야 제대로 하나의 bar가 하나의 단위가 된다.
                csvarray.append(row)
        csvarray = np.array(csvarray)
        if ('timeSignature' not in jsonlist[i]):
            jsonlist[i]['timeSignature'] = [4, 4]
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


def bar_to_matrix1(bar, one_bar_number, starting_number, i):
    # bar_to_matrix
    # 8/6박이면 one_bar_number가 8이다. 그러면 바 1개당 무조건 12개 처리하는거로 한다.
    # lists[0]은 시간, lists[1]은 Note 높이, lists[3]은 Duration.
    init = np.zeros((112, 96))  # 112 is number of note pitch.(not precise value, but it can handle all notes), 96 is note timing.
    minimum_size = one_bar_number / 96
    zero_time = starting_number + one_bar_number * i
    for j, lists in enumerate(bar):
        point = int((nearest_time(lists[0], minimum_size) - zero_time) / minimum_size)
        if (point >= 96):# error handling if one_bar_number is not 3,4,6,8..like.(dataset contains 5/4)
            point = 92

        init[111 - int(lists[1])][point] = lists[3]  # 111-int(list[1])형태로 해야 직관적인 PianoRoll 형태가 ㅏ온다.
    return init


def bar_to_matrix2(bar, one_bar_number, starting_number, i):
    # Duration에 따라 Ploting한다.
    # 8/6박이면 one_bar_number가 8이다. 그러면 바 1개당 무조건 12개 처리하는거로 한다.

    init = np.zeros((112, 96))  # 112는 Note의 수(감으로 써둠.. 나중에 전체 데이터로 할때 수정 가능성 있음 그런데 Shift를 잘 이용하면 96*96도 가능해보임.)
    minimum_size = one_bar_number / 96
    zero_time = starting_number + one_bar_number * i
    for j, lists in enumerate(bar):
        # lists[0]은 시간, lists[1]은 Note 높이, lists[3]은 Duration.
        point = int((nearest_time(lists[0], minimum_size) - zero_time) / minimum_size)
        length = int(round(lists[3] / minimum_size))
        if (length > 3):
            length = length - 1  # 여러번 두두두 치는 음을 구별하기 위함
        if (point + length > 95):
            length = 95 - point  # 한 음이 2Bar에 걸쳐있는 경우 Bar 뒤쪽의 음을 무시한다.
        init[111 - int(lists[1])][point:point + length] += 1
    return init


def bar_to_matrix3(bar, one_bar_number, starting_number, i):
    # 일단 3은 쓰지 맙시다
    # Duration에 따라 Plot함과 동시에 가장 높은음과 낮은음을 고려해서 실제 DAW와 같이 Plot한다.
    # 8/6박이면 one_bar_number가 8이다. 그러면 바 1개당 무조건 12개 처리하는거로 한다.
    init = np.zeros((112, 96))  # 112는 Note의 수(감으로 써둠.. 나중에 전체 데이터로 할때 수정 가능성 있음 그런데 Shift를 잘 이용하면 96*96도 가능해보임.)
    minimum_size = one_bar_number / 96
    zero_time = starting_number + one_bar_number * i
    for j, lists in enumerate(bar):
        # lists[0]은 시간, lists[1]은 Note 높이, lists[3]은 Duration.
        point = int((nearest_time(lists[0], minimum_size) - zero_time) / minimum_size)
        init[int(lists[1])][point] = lists[3]
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