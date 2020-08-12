import matplotlib
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from keras import utils as keras_utils
from keras import datasets
from swiss_army_tensorboard import tfboard_loggers
import keras
import os
import pandas
import numpy as np
from keras import optimizers
from cdcgan import cdcgan_models, cdcgan_utils
from keras.callbacks import ModelCheckpoint
import copy
import pretty_midi
import mido
from tqdm import tqdm
from model import f1score,precision,make_classifier,make_model,recall
from utils import get_meta,csv_to_array,midi_to_array,bar_to_matrix2,set_labels,get_tag_results
from feature_defining import bar_to_contour, contour_to_label
from sklearn.preprocessing import MultiLabelBinarizer
from plotting import plot_recall_f1score, plot_val_loss

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
labels=set_labels()
mlb.fit(labels)
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
hist=classifier.fit(
      train_matrix,train_label2,batch_size=32,
      epochs=10,
      validation_data=(valid_matrix,valid_label2)
  ,verbose=2, callbacks=callbacks
)
#plotting
plot_val_loss(hist)
plot_recall_f1score(hist)
# testing

testresult=classifier.predict(test_matrix)

matplotlib.use('Agg')
best=get_tag_results(testresult,test_label2)[0]
print(best)


BATCH_SIZE = 128
EPOCHS = 100

# Load & Prepare MNIST
X_train=train_matrix.reshape((1500,112,96))
testresult=classifier.predict(train_matrix)
y_train=np.array(get_tag_results(testresult,train_label2)[0]).reshape((1500,))
X_train = cdcgan_utils.transform_images(X_train)
X_train = X_train[:, :, :, None]



encoder = LabelEncoder()

# X_train데이터를 이용 피팅하고 라벨숫자로 변환한다
encoder.fit(y_train)
y_train = encoder.transform(y_train)
print(y_train)
y_train=np.array(y_train)

# Create the models
y_train = keras_utils.to_categorical(y_train, 100)

print("Generator:")
G = cdcgan_models.generator_model()
G.summary()

print("Discriminator:")
D = cdcgan_models.discriminator_model()
D.summary()

print("Combined:")
GD = cdcgan_models.generator_containing_discriminator(G, D)
GD.summary()

optimizer = optimizers.Adam(0.0002, 0.5)

G.compile(loss='binary_crossentropy', optimizer=optimizer)
GD.compile(loss='binary_crossentropy', optimizer=optimizer)
D.trainable = True
D.compile(loss='binary_crossentropy', optimizer=optimizer)

# Setup Tensorboard loggers

tfboard_loggers.TFBoardModelGraphLogger.log_graph("../models/logs", K.get_session())
loss_logger = tfboard_loggers.TFBoardScalarLogger("../models/logs/loss")
image_logger = tfboard_loggers.TFBoardImageLogger("../models/logs/generated_images")

# Model Training

iteration = 0

nb_of_iterations_per_epoch = int(X_train.shape[0] / BATCH_SIZE)
print("Number of iterations per epoch: {0}".format(nb_of_iterations_per_epoch))

for epoch in range(EPOCHS):
    pbar = tqdm(desc="Epoch: {0}".format(epoch), total=X_train.shape[0])

    g_losses_for_epoch = []
    d_losses_for_epoch = []

    for i in range(nb_of_iterations_per_epoch):
        noise = cdcgan_utils.generate_noise((BATCH_SIZE, 100))

        image_batch = X_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        label_batch = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        generated_images = G.predict([noise, label_batch], verbose=0)

        if i % 20 == 0:
            image_grid = cdcgan_utils.generate_mnist_image_grid(G,
                                                                title="Epoch {0}, iteration {1}".format(epoch,
                                                                                                        iteration))
            cdcgan_utils.save_generated_image(image_grid, epoch, i, "../images/generated_mnist_images_per_iteration")
            image_logger.log_images("generated_mnist_images_per_iteration", [image_grid], iteration)

        X = np.concatenate((image_batch, generated_images))
        y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
        label_batches_for_discriminator = np.concatenate((label_batch, label_batch))

        D_loss = D.train_on_batch([X, label_batches_for_discriminator], y)
        d_losses_for_epoch.append(D_loss)
        loss_logger.log_scalar("discriminator_loss", D_loss, iteration)

        noise = cdcgan_utils.generate_noise((BATCH_SIZE, 100))
        D.trainable = False
        G_loss = GD.train_on_batch([noise, label_batch], [1] * BATCH_SIZE)
        D.trainable = True
        g_losses_for_epoch.append(G_loss)
        loss_logger.log_scalar("generator_loss", G_loss, iteration)

        pbar.update(BATCH_SIZE)

        iteration += 1

    # Save a generated image for every epoch
    image_grid = cdcgan_utils.generate_mnist_image_grid(G, title="Epoch {0}".format(epoch))
    cdcgan_utils.save_generated_image(image_grid, epoch, 0, "../images/generated_mnist_images_per_epoch")
    image_logger.log_images("generated_mnist_images_per_epoch", [image_grid], epoch)

    pbar.close()
    print("D loss: {0}, G loss: {1}".format(np.mean(d_losses_for_epoch), np.mean(g_losses_for_epoch)))

    G.save_weights("../models/weights/generator.h5")
    D.save_weights("../models/weights/discriminator.h5")
