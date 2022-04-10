from tensorflow.python.client import device_lib
from util import *
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from hyperparameter import Hyperparams as hp
import numpy as np
import keras.applications
from keras import regularizers, optimizers
from keras import layers
from tensorflow import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers, models
import imageio
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from keras.utils import np_utils
import warnings
from io import BytesIO
from pathlib import Path
from typing import Union, List
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gan_util import *

# Original Codes for Logger and GAN model implemented by gaborvecsei, https://github.com/gaborvecsei.
class _BaseTFBoardLogger:
    def __init__(self, log_dir: Union[Path, str]):
        log_dir = str(log_dir)

        if Path(log_dir).is_dir():
            warnings.warn("Folder {0} is already created, maybe it contains other log files".format(log_dir))

        self._log_dir = log_dir
        self._summary_writer = tf.summary.create_file_writer(log_dir)

class TFBoardImageLogger(_BaseTFBoardLogger):
    def __init__(self, log_dir: Union[Path, str]):
        super().__init__(log_dir)

    def log_images(self, tag: str, images: Union[List[np.ndarray], np.ndarray], step: int):
        image_summaries = []
        for i, image in enumerate(images):
            image_str = BytesIO()
            plt.imsave(image_str, image, format='png')
            height, width = image.shape[:2]
            image_summary = tf.Summary.Image(encoded_image_string=image_str.getvalue(),
                                             height=height,
                                             width=width)
            image_tag = "{0}/{1}".format(tag, i)
            image_summaries.append(tf.Summary.Value(tag=image_tag, image=image_summary))

        summary = tf.Summary(value=image_summaries)
        self._summary_writer.add_summary(summary, step)

class GAN_models(object):
    def __init__(self):
        self.minimum_time = hp.Minimum_time
        self.all_matrix = np.load('preprocessing/all_matrix.npy', allow_pickle=True)
        self.all_labels = np.load('preprocessing/all_labels.npy', allow_pickle=True)
        self.all_primining_notes = np.load('preprocessing/all_primining_notes.npy', allow_pickle=True)
        classifier = classifier_models()
        self.classifier = classifier.make_label_model()
        self.classifier.load_weights("models/label_classifier.h5")#loading for best classifier model
        self.epoch = hp.GAN_epochs
        self.batch_size = hp.GAN_BATCH_SIZE
        self.learning_rate = hp.GAN_learning_rate

    def get_label(self):
        trainX=self.all_matrix.reshape((int(len(self.all_matrix)),24,self.minimum_time))
        blur_trainX=copy.deepcopy(trainX)
        for i,matrix in enumerate(trainX):
            blur_trainX[i]=blur_image(matrix)
        trainX=blur_trainX


        for i in tqdm(range(int(len(self.all_matrix)//1000)),position=0):
            if(i==0):
                test_result=self.classifier.predict(self.all_matrix[1000*i:1000+1000*i])
            else:
                sub_testresult=self.classifier.predict(self.all_matrix[1000*i:1000+1000*i])
                test_result=np.concatenate((test_result,sub_testresult))
        sub_testresult=self.classifier.predict(self.all_matrix[1000*(i+1):])
        test_result=np.concatenate((test_result,sub_testresult))
        tot_dict={}
        mlb=MultiLabelBinarizer()
        labels=set_labels()
        mlb.fit(labels)
        all_label = mlb.transform(self.all_labels)
        trainy=np.array(get_best_results(test_result,all_label)).reshape((int(len(self.all_matrix)),))
        encoder = LabelEncoder()
        encoder.fit(trainy)
        trainy = encoder.transform(trainy)
        trainy=np.array(trainy)
        """
        for results in trainy:
            if str(results) not in tot_dict:
                tot_dict[str(results)]=1
            else:
                tot_dict[str(results)]+=1
        print(tot_dict)
        """
        return trainX, trainy

    def generator_model(self,minimum_time):
        ACTIVATION = layers.Activation("tanh")
        # Prepare noise input
        input_z = layers.Input((hp.Label_num,))
        dense_z_1 = layers.Dense(1024)(input_z)
        act_z_1 = ACTIVATION(dense_z_1)
        dense_z_2 = layers.Dense(128 * 6 * int(minimum_time/4))(act_z_1)
        bn_z_1 = layers.BatchNormalization()(dense_z_2)
        reshape_z = layers.Reshape((6, int(minimum_time/4), 128), input_shape=(128 * 6 * int(minimum_time/4),))(bn_z_1) # 6, 4, 128

        # Prepare Conditional (label) input
        input_c = layers.Input((hp.Label_num,))
        dense_c_1 = layers.Dense(1024)(input_c)
        act_c_1 = ACTIVATION(dense_c_1)
        dense_c_2 = layers.Dense(128 * 6 * int(minimum_time/4))(act_c_1)
        bn_c_1 = layers.BatchNormalization()(dense_c_2)
        reshape_c = layers.Reshape((6, int(minimum_time/4), 128), input_shape=(128 * 6 * int(minimum_time/4),))(bn_c_1) # 6, 4, 128


        # Combine input source
        concat_z_c = layers.Concatenate()([reshape_z, reshape_c])# 6, 4, 384
        #multiple_z_c = layers.Multiply()([reshape_z, reshape_c])#6, 4, 192

        # Prepare Conditional (Midi image) input

        input_time_v = layers.Input((128,))
        input_pitch_v = layers.Input((128,))

        dense_input_time_v_1 = layers.Dense(64)(input_time_v)
        act_input_time_1 = ACTIVATION(dense_input_time_v_1)
        dense_input_time_v_2 = layers.Dense(8)(act_input_time_1)
        dense_input_time_v_2 = layers.Reshape((8,1), input_shape=(8,))(dense_input_time_v_2)

        dense_input_pitch_v_1 = layers.Dense(64)(input_pitch_v)
        act_input_pitch_1 = ACTIVATION(dense_input_pitch_v_1)
        dense_input_pitch_v_2 = layers.Dense(12)(act_input_pitch_1)

        transpose_input_pitch_v = layers.Reshape((1,12), input_shape = (12,))(dense_input_pitch_v_2)

        projection_layer = layers.Dot(axes=(1,2))([transpose_input_pitch_v, dense_input_time_v_2]) # 12 by 8 expected
        projection_layer = layers.Reshape((12,8,1), input_shape = (12,8))(projection_layer) # simply make 1 virtual channel

        projection_conv_layer = layers.Conv2D(256,(2,2), padding='same')(projection_layer)
        bn_projection_layer = layers.BatchNormalization()(projection_conv_layer)

        # Image generation with the concatenated inputs
        up_1 = layers.UpSampling2D(size=(2, 2))(concat_z_c) # 12, 8, N
        up_1 = layers.BatchNormalization()(up_1)#for adding layer

        projection_1 =  layers.Concatenate()([up_1,bn_projection_layer]) 

        conv_1 = layers.Conv2D(64, (5, 5), padding='same')(projection_1) # 12, 8, 64
        act_1 = ACTIVATION(conv_1)
        up_2 = layers.UpSampling2D(size=(2, 2))(act_1) # 24, 16, 64
        conv_2 = layers.Conv2D(1, (5, 5), padding='same')(up_2)
        act_2 = layers.Activation("tanh")(conv_2)
        model = models.Model(inputs=[input_z, input_c, input_time_v, input_pitch_v], outputs=act_2)
        return model

    def discriminator_model(self,minimum_time):
        ACTIVATION = layers.Activation("tanh")
        input_gen_image = layers.Input((24, minimum_time, 1))
        conv_1_image = layers.Conv2D(64, (5, 5), padding='same')(input_gen_image)
        act_1_image = ACTIVATION(conv_1_image)
        pool_1_image = layers.MaxPooling2D(pool_size=(2, 2))(act_1_image)
        conv_2_image = layers.Conv2D(128, (5, 5))(pool_1_image)
        act_2_image = ACTIVATION(conv_2_image)
        pool_2_image = layers.MaxPooling2D(pool_size=(2, 2))(act_2_image)

        input_c = layers.Input((hp.Label_num,))
        dense_1_c = layers.Dense(1024)(input_c)
        act_1_c = ACTIVATION(dense_1_c)
        dense_2_c = layers.Dense(4 * int(minimum_time/4-2) * 128)(act_1_c)
        bn_c = layers.BatchNormalization()(dense_2_c)
        reshaped_c = layers.Reshape((4, int(minimum_time/4-2), 128))(bn_c) # 4, 2, 128

        input_time_v = layers.Input((128,))
        input_pitch_v = layers.Input((128,))

        dense_input_time_v_1 = layers.Dense(64)(input_time_v)
        act_input_time_1 = ACTIVATION(dense_input_time_v_1)
        dense_input_time_v_2 = layers.Dense(2)(act_input_time_1)
        dense_input_time_v_2 = layers.Reshape((2,1), input_shape=(8,))(dense_input_time_v_2)

        dense_input_pitch_v_1 = layers.Dense(64)(input_pitch_v)
        act_input_pitch_1 = ACTIVATION(dense_input_pitch_v_1)
        dense_input_pitch_v_2 = layers.Dense(4)(act_input_pitch_1)

        transpose_input_pitch_v = layers.Reshape((1,4), input_shape = (4,))(dense_input_pitch_v_2)

        projection_layer = layers.Dot(axes=(1,2))([transpose_input_pitch_v, dense_input_time_v_2]) # 12 by 8 expected
        projection_layer = layers.Reshape((4,2,1), input_shape = (4,2))(projection_layer) # simply make 1 virtual channel

        projection_conv_layer = layers.Conv2D(256,(2,2), padding='same')(projection_layer)
        bn_projection_layer = layers.BatchNormalization()(projection_conv_layer)


        concat = layers.Concatenate()([pool_2_image, reshaped_c])
        #multiple = layers.Multiply()([pool_2_image, reshaped_c])
        #concat = layers.Add()([concat, bn_projection_layer])
        concat = layers.BatchNormalization()(concat)
        concat = layers.Concatenate()([concat, bn_projection_layer])
        flat = layers.Flatten()(concat)
        dense_1 = layers.Dense(1024)(flat)
        
        act_1 = ACTIVATION(dense_1)
        dense_2 = layers.Dense(1)(act_1)
        act_2 = layers.Activation('sigmoid')(dense_2)
        model = models.Model(inputs=[input_gen_image, input_c, input_time_v, input_pitch_v], outputs=act_2)
        return model
    
    def generator_containing_discriminator(self,g, d, minimum_time):
        input_z = layers.Input((hp.Label_num,))
        input_c = layers.Input((hp.Label_num,))
        input_time_v = layers.Input((128,))
        input_pitch_v = layers.Input((128,))
        input_target = layers.Input((24, minimum_time, 1))
        gen_image = g([input_z, input_c, input_time_v, input_pitch_v])
        d.trainable = False
        is_real = d([gen_image, input_c, input_time_v, input_pitch_v])
        model = models.Model(inputs=[input_z, input_c, input_pitch_v, input_time_v], outputs=[is_real, gen_image])
        return model


    def training_gan(self):
        X_train, y_train = self.get_label()

        X_train = transform_images(X_train)
        X_train = X_train[:, :, :, None]
        V_train = self.all_primining_notes # (54051, 2, N), with Various length N. V_train[0][0] -> rel pitch of First batch
        plt.imshow(X_train[1].reshape((24,16)))
        plt.savefig('datatest.png')

        y_train = np_utils.to_categorical(y_train, hp.Label_num)
        rel_pitch_train = tf.keras.preprocessing.sequence.pad_sequences(V_train[:,0],maxlen=128)
        rel_time_train = tf.keras.preprocessing.sequence.pad_sequences(V_train[:,1], maxlen=128) # padding with maximum length 128.

        # Create the models

        G = self.generator_model(self.minimum_time)
        D = self.discriminator_model(self.minimum_time)

        GD = self.generator_containing_discriminator(G, D,self.minimum_time)

        optimizer = tf.keras.optimizers.Adam(self.learning_rate, 0.5)

        G.compile(loss='binary_crossentropy',  optimizer=optimizer)
        GD.compile(loss=['binary_crossentropy',tf.keras.losses.MeanSquaredError()], loss_weights = [1,0.01], optimizer=optimizer)
        
        D.trainable = True
        D.compile(loss='binary_crossentropy',loss_weights=[0.1], optimizer=optimizer)
        """
        tf.keras.utils.plot_model(G,to_file='generator.png')
        tf.keras.utils.plot_model(D,to_file='discriminator.png')
        """
        # Model Training

        image_logger = TFBoardImageLogger("GAN_result/logs/generated_images")

        EPOCH = self.epoch
        BATCH_SIZE = self.batch_size

        iteration = 0
        nb_of_iterations_per_epoch = int(X_train.shape[0] / BATCH_SIZE)
        print("Number of iterations per epoch: {0}".format(nb_of_iterations_per_epoch))

        for epoch in range(EPOCH):
            pbar = tqdm(desc="Epoch: {0}".format(epoch), total=X_train.shape[0],position=0)
            dist_val = 0.8
            g_losses_for_epoch = []
            d_losses_for_epoch = []

            for i in range(nb_of_iterations_per_epoch):
                noise = generate_noise((BATCH_SIZE, hp.Label_num))

                image_batch = np.array(X_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
                label_batch = np.array(y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
                rel_pitch_batch = np.array(rel_pitch_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
                rel_time_batch = np.array(rel_time_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])

                generated_images = G.predict([noise, label_batch,rel_time_batch,rel_pitch_batch], verbose=0)
                X = np.concatenate((image_batch, generated_images))
                y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
                label_batches_for_discriminator = np.concatenate((label_batch, label_batch))
                time_vector_condition_for_discriminator = np.concatenate((rel_time_batch,rel_time_batch))
                pitch_vector_condition_for_discriminator = np.concatenate((rel_pitch_batch,rel_pitch_batch))
                
                if(np.random.random()<dist_val):
                    D_loss = D.train_on_batch([X, label_batches_for_discriminator, time_vector_condition_for_discriminator,pitch_vector_condition_for_discriminator], y)
                    d_losses_for_epoch.append(D_loss)

                noise = generate_noise((BATCH_SIZE, hp.Label_num))
                D.trainable = False
                G_loss = GD.train_on_batch([noise, label_batch,rel_time_batch,rel_pitch_batch], [np.array([1] * BATCH_SIZE),image_batch])
                D.trainable = True
                g_losses_for_epoch.append(G_loss)

                pbar.update(BATCH_SIZE)

                iteration += 1
                
            # Save a generated image for every epoch
            image_grid = generate_mnist_image_grid(G, rel_time_batch, rel_pitch_batch, title="Epoch {0}".format(epoch)) 
            save_generated_image(image_grid, epoch, 0, "GAN_result/generated_mnist_images_per_epoch")

            pbar.close()
            print("D loss: {0}, G loss: {1}".format(np.mean(d_losses_for_epoch), np.mean(g_losses_for_epoch)))
            if(np.mean(g_losses_for_epoch)>1):
                dist_val=0.4 # discriminator batch trains with probability 0.4 for balance between G and D.
            else:
                dist_val=0.9


            G.save_weights("models/generator.h5")
            D.save_weights("models/discriminator.h5")
        
        
class RNN_models(object):
    def __init__(self):
        self.minimum_time = hp.Minimum_time
        self.all_matrix = np.load('preprocessing/all_matrix.npy', allow_pickle=True)
        self.all_labels = np.load('preprocessing/all_labels.npy', allow_pickle=True)
        self.all_updown_labels = np.load('preprocessing/all_updown_labels.npy', allow_pickle=True)
        self.bar_matrix_lists = np.load('preprocessing/bar_matrix_lists.npy', allow_pickle=True)

        self.train_matrix=np.array(self.all_matrix[:int(len(self.all_matrix)*0.85)])
        self.train_label=np.array(self.all_labels[:int(len(self.all_matrix)*0.85)])
        self.train_updown_label=np.array(self.all_updown_labels[:int(len(self.all_matrix)*0.85)])

        self.valid_matrix=np.array(self.all_matrix[int(len(self.all_matrix)*0.85):int(len(self.all_matrix)*0.95)])
        self.valid_label=np.array(self.all_labels[int(len(self.all_matrix)*0.85):int(len(self.all_matrix)*0.95)])
        self.valid_updown_label=np.array(self.all_updown_labels[int(len(self.all_matrix)*0.85):int(len(self.all_matrix)*0.95)])

        self.test_matrix=np.array(self.all_matrix[int(len(self.all_matrix)*0.95):])
        self.test_label=np.array(self.all_labels[int(len(self.all_matrix)*0.95):])
        self.test_updown_label=np.array(self.all_updown_labels[int(len(self.all_matrix)*0.95):])

        classifier = classifier_models()
        self.classifier = classifier.make_label_model()
        self.classifier.load_weights("models/label_classifier.h5")#loading for best classifier model

    def make_RNNmodel(self):
        embedding_vector_length = 20
        timesteps = 8
        top_words=hp.Label_num
        RNNmodel = Sequential()
        RNNmodel.add(Embedding(top_words, embedding_vector_length, input_length=20))
        RNNmodel.add(GRU(100, return_sequences=True,
                    input_shape=(timesteps, 100)))  # returns a sequence of vectors of dimension 32
        RNNmodel.add(LSTM(100))  # return a single vector of dimension 32
        RNNmodel.add(Dropout(0.2))
        RNNmodel.add(Dense(hp.Label_num, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
        RNNmodel.summary()
        RNNmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return RNNmodel

    def RNN_train(self):

        RNNx_train=[]
        RNNy_train=[]
        RNNx_test=[]
        RNNy_test=[]
        RNN_kfoldx=[]
        RNN_kfoldy=[]
        for i in range(len(self.bar_matrix_lists)):
            nowseq=[]
            nowmat=np.array(self.bar_matrix_lists[i])
            nowbars=self.classifier.predict(nowmat.reshape(len(nowmat),24,self.minimum_time,1) )
            for j,bars in enumerate(nowbars):
                if(j==len(nowbars)-1):
                    if(i>=len(self.bar_matrix_lists)*0.85):
                        RNNy_test.append(np.argmax(bars))
                    else:
                        RNNy_train.append(np.argmax(bars))
                    RNN_kfoldy.append(np.argmax(bars))
                else:
                    nowseq.append(np.argmax(bars)+1)
            if(i>=len(self.bar_matrix_lists)*0.85):
                RNNx_test.append(nowseq)
            else:
                RNNx_train.append(nowseq)
            RNN_kfoldx.append(nowseq)
        RNNy_test=to_categorical(np.array(RNNy_test))
        RNNy_train=to_categorical(np.array(RNNy_train))
        RNN_kfoldy=to_categorical(np.array(RNN_kfoldy))
        RNNx_train = sequence.pad_sequences(RNNx_train, maxlen=20)
        RNNx_test = sequence.pad_sequences(RNNx_test, maxlen=20)
        RNN_kfoldx = sequence.pad_sequences(RNN_kfoldx, maxlen=20)

        # fix random seed for reproducibility
        np.random.seed(7)
        unique, counts = np.unique(RNN_kfoldx, return_counts=True)
        RNN_weight=dict(zip(unique, 5000/counts))

        model_path = 'models/RNN.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy',
                                        verbose=1, save_best_only=True)
        callbacks = [cb_checkpoint]
        RNNmodel = self.make_RNNmodel()

        kfold = KFold(n_splits=5, shuffle=True, random_state=7)
        for i in range(3):
            for train,valid in kfold.split(RNN_kfoldx,RNN_kfoldy):
                RNNmodel.fit(RNN_kfoldx[train],RNN_kfoldy[train], validation_data=(RNN_kfoldx[valid],RNN_kfoldy[valid])
                ,epochs=15, batch_size=32,callbacks=callbacks)

        """
        RNNmodel.fit(RNN_kfoldx[:len(RNN_kfoldx*0.8)],RNN_kfoldy[:len(RNN_kfoldx*0.8)],
         validation_data=(RNN_kfoldx[len(RNN_kfoldx*0.8):], RNN_kfoldy[len(RNN_kfoldx*0.8):]),
         epochs=500, batch_size=32, callbacks = callbacks)
        """

class classifier_models(object):
    def __init__(self):

        self.minimum_time = hp.Minimum_time
        self.all_matrix = np.load('preprocessing/all_matrix.npy', allow_pickle=True)
        self.all_labels = np.load('preprocessing/all_labels.npy', allow_pickle=True)
        self.all_updown_labels = np.load('preprocessing/all_updown_labels.npy', allow_pickle=True)

        self.train_matrix=np.array(self.all_matrix[:int(len(self.all_matrix)*0.85)])
        self.train_label=np.array(self.all_labels[:int(len(self.all_matrix)*0.85)])
        self.train_updown_label=np.array(self.all_updown_labels[:int(len(self.all_matrix)*0.85)])

        self.valid_matrix=np.array(self.all_matrix[int(len(self.all_matrix)*0.85):int(len(self.all_matrix)*0.95)])
        self.valid_label=np.array(self.all_labels[int(len(self.all_matrix)*0.85):int(len(self.all_matrix)*0.95)])
        self.valid_updown_label=np.array(self.all_updown_labels[int(len(self.all_matrix)*0.85):int(len(self.all_matrix)*0.95)])

        self.test_matrix=np.array(self.all_matrix[int(len(self.all_matrix)*0.95):])
        self.test_label=np.array(self.all_labels[int(len(self.all_matrix)*0.95):])
        self.test_updown_label=np.array(self.all_updown_labels[int(len(self.all_matrix)*0.95):])

    def label_classifier_train(self):
        print("label_classifier_training")

        mlb=MultiLabelBinarizer()
        labels=set_labels()
        mlb.fit(labels)
        train_label2=mlb.transform(self.train_label)
        valid_label2=mlb.transform(self.valid_label)
        test_label2=mlb.transform(self.test_label)
        all_label = mlb.transform(self.all_labels)
        tot_array = np.zeros(hp.Label_num)
        for label in all_label:
            tot_array += label
        tot_array = 1 / tot_array
        tot_array = tot_array /  np.linalg.norm(tot_array)
        class_weights={}
        for idx, weights in enumerate(tot_array):
            class_weights[idx]=weights

        model_path = 'models/label_classifier.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy',
                                verbose=1, save_best_only=True)
        callbacks = [cb_checkpoint]

        classifier = self.make_label_model()
        classifier.compile(loss=keras.losses.BinaryCrossentropy(
            from_logits=False, label_smoothing=0.1, 
            name='binary_crossentropy'
        ), optimizer='adam', metrics=['accuracy',recall,precision,f1score])

        hist=classifier.fit(
            self.train_matrix,train_label2,batch_size=256,
            epochs=100,
            validation_data=(self.valid_matrix,valid_label2),
            callbacks=callbacks,class_weight=class_weights)

    def updown_classifier_train(self):
        print("updown_classifier_training")
        model_path = 'models/updown.h5'
        updownle=LabelBinarizer()
        updownle.fit(['up','down','final','meanless'])
        train_updown_label=updownle.transform(np.array(self.train_updown_label))
        valid_updown_label=updownle.transform(np.array(self.valid_updown_label))
        test_updown_label=updownle.transform(np.array(self.test_updown_label))

        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy',
                                verbose=1, save_best_only=True)
        updown_callbacks=[cb_checkpoint]
        updown_classifier=self.make_updown_classifier()
        updown_classifier.compile(loss=keras.losses.BinaryCrossentropy(
      from_logits=False,
      name='binary_crossentorpy',), optimizer='adam', metrics=['accuracy'])

        updown_hist=updown_classifier.fit(
      self.train_matrix,train_updown_label,batch_size=256,
      epochs=50,
      validation_data=(self.valid_matrix,valid_updown_label),
      callbacks=updown_callbacks,)

    def residual_block(self,filter, input, add=True):
        with tf.device('/gpu:0'):
            layer_1 = keras.layers.Conv2D(filters=filter//4, kernel_size=(1, 1), data_format="channels_first")(input)
            layer_2 = keras.layers.Conv2D(filters=filter//4, kernel_size=(3, 3), padding='same', data_format="channels_first", kernel_regularizer=keras.regularizers.l2(0.001))(layer_1)
            layer_2 = keras.layers.BatchNormalization()(layer_2)
            layer_2 = keras.layers.ReLU()(layer_2)
            layer_3 = keras.layers.Conv2D(filters=filter, kernel_size=(1, 1), data_format="channels_first")(layer_2)
            layer_3 = keras.layers.BatchNormalization()(layer_3)
            if add:
                layer_3 = keras.layers.add([input, layer_3])
            layer_3 = keras.layers.ReLU()(layer_3)
            return layer_3
    def make_label_model(self):
        with tf.device('/gpu:0'):
            input_layer = keras.Input(shape=(24, self.minimum_time, 1))
            layer_1 = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), padding='same', data_format="channels_first")(input_layer)
            block_1 = self.residual_block(64, layer_1)
            #block_2 = residual_block(64, block_1)
            #block_3 = residual_block(64, block_2)
            pooling_layer = keras.layers.MaxPool2D((2, 2),padding='same', data_format="channels_first")(block_1)
            block_4 = self.residual_block(128, pooling_layer, add=False)
            block_5 = self.residual_block(128, block_4)
            #block_6 = residual_block(128, block_5)
            pooling_layer2 = keras.layers.MaxPool2D(padding='same',pool_size=(2, 2), data_format="channels_first")(block_4)
            block_7 = self.residual_block(256, pooling_layer2, add=False)
            block_8 = self.residual_block(256, block_7)
            block_9 = self.residual_block(256, block_8)
            #pooling_layer4 = keras.layers.MaxPool2D(pool_size=(2, 2), data_format="channels_first")(block_7)
            #block_10 = residual_block(256, pooling_layer4)
            #block_11 = residual_block(512, block_10)
            pooling_layer3 = keras.layers.AvgPool2D(padding='same',pool_size=(8, 8), data_format="channels_first")(block_7)
            last_layer = keras.layers.Flatten()(pooling_layer3)
            last_layer = keras.layers.Dropout(0.4)(last_layer)
            last_layer = keras.layers.Dense(hp.Label_num, activation="sigmoid")(last_layer)
            return keras.models.Model(inputs=input_layer, outputs=last_layer)

    def make_updown_classifier(self):
        with tf.device('/gpu:0'):
            classifier = keras.Sequential()
            classifier.add(keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same',
                        activation='relu',
                        input_shape=(24,self.minimum_time,1)))
            classifier.add(keras.layers.BatchNormalization())
            classifier.add(keras.layers.LeakyReLU(alpha=0.01))
            classifier.add(keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same'))
            classifier.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            classifier.add(keras.layers.Conv2D(256, (2, 2), padding='same'))
            classifier.add(keras.layers.LeakyReLU(alpha=0.01))
            classifier.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
            classifier.add(keras.layers.Flatten())
            classifier.add(keras.layers.Dropout(0.25))
            classifier.add(keras.layers.Dense(4, activation='sigmoid'))
        return classifier    
        
        
    