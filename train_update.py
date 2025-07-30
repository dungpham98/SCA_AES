import os
import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import ast
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, BatchNormalization, Activation, Add, add
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import os.path
import sys
import h5py
import math
import gc
import time
import numpy as np
#from numba import cuda
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, ReLU, Dropout, Concatenate, Activation, Multiply, BatchNormalization #, AveragePooling1D, Add, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.utils import plot_model   #, get_source_inputs
from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K
#from keras_applications.imagenet_utils import decode_predictions
#from keras_applications.imagenet_utils import preprocess_input
#from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
#from tensorflow.python.keras.layers import Lambda
#from sklearn.model_selection import train_test_split
#K-center: https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
# Trace and metadata parameters
from pathlib import Path
from sklearn.cluster import KMeans

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

set_seeds(2025)

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
    parser.add_argument('--num_epoch', type=int, help='batch_size', default=256)
    parser.add_argument('--num_sample', type=int, help='batch_size', default=256)
    parser.add_argument('--num_it', type=int, help='batch_size', default=10)
    parser.add_argument('--sampling', type=str, default='None')
    parser.add_argument('--eval_interval', type=int, help='iteration_num', default=1)
    parser.add_argument('--name', type=str, help='experiment name', default='test')
    parser.add_argument('--train_folder', type=str)
    parser.add_argument('--var', type=int, help='use variable key', default=0)

    return parser   

def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

def load_ascad(ascad_database_file, mask_type = 'MS1'):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file  = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    device_name =  [key for key in in_file.keys()][0]
    # Load profiling traces
    X_profiling = np.array(in_file['{}/{}/Profiling/Traces'.format(device_name, mask_type)], dtype=np.int8)
    # Load profiling labels
    Y_profiling = np.array(in_file['{}/{}/Profiling/Labels'.format(device_name, mask_type)])
    # Load attacking traces
    X_attack = np.array(in_file['{}/{}/Attack/Traces'.format(device_name, mask_type)], dtype=np.int8)
    # Load attacking labels
    Y_attack = np.array(in_file['{}/{}/Attack/Labels'.format(device_name, mask_type)])

    return (X_profiling, Y_profiling), (X_attack, Y_attack)

def load_multi_attack(data_path):
    infile = np.load(data_path)
    data = infile['data']
    labels = infile['label']

    return data, labels

### CNN Best model
def cnn_best(classes=256,input_dim=700):
    # From VGG16 design
    input_shape = (input_dim,1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_best')
    optimizer = RMSprop(learning_rate=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def mlp_best(node=200,layer_nb=6,input_dim=1400):
    model = Sequential()
    model.add(Dense(node, input_dim=input_dim, activation='relu'))
    #model.add(BatchNormalization(node))
    model.add(BatchNormalization())
    for i in range(layer_nb-2):
            model.add(Dense(node, activation='relu'))
            model.add(BatchNormalization())
    model.add(Dense(256, activation='softmax'))
    optimizer = RMSprop(learning_rate=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
#TODO: Train/Val split
#Train baseline

# The AES SBox that we will use to compute the rank
AES_Sbox = np.array([
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
        ])

# Two Tables to process a field multplication over GF(256): a*b = alog (log(a) + log(b) mod 255)
log_table=[ 0, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3,
    100, 4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193,
    125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9, 120,
    101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218, 142,
    150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70, 131, 56,
    102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34, 136, 145, 16,
    126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84, 250, 133, 61, 186,
    43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172, 229, 243, 115, 167, 87,
    175, 88, 168, 80, 244, 234, 214, 116, 79, 174, 233, 213, 231, 230, 173, 232,
    44, 215, 117, 122, 235, 22, 11, 245, 89, 203, 95, 176, 156, 169, 81, 160,
    127, 12, 246, 111, 23, 196, 73, 236, 216, 67, 31, 45, 164, 118, 123, 183,
    204, 187, 62, 90, 251, 96, 177, 134, 59, 82, 161, 108, 170, 85, 41, 157,
    151, 178, 135, 144, 97, 190, 220, 252, 188, 149, 207, 205, 55, 63, 91, 209,
    83, 57, 132, 60, 65, 162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171,
    68, 17, 146, 217, 35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165,
    103, 74, 237, 222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7 ]

alog_table =[1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53,
    95, 225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170,
    229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230, 49,
    83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110, 178, 205,
    76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24, 40, 120, 136,
    131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206, 73, 219, 118, 154,
    181, 196, 87, 249, 16, 48, 80, 240, 11, 29, 39, 105, 187, 214, 97, 163,
    254, 25, 43, 125, 135, 146, 173, 236, 47, 113, 147, 174, 233, 32, 96, 160,
    251, 22, 58, 78, 210, 109, 183, 194, 93, 231, 50, 86, 250, 21, 63, 65,
    195, 94, 226, 61, 71, 201, 64, 192, 91, 237, 44, 116, 156, 191, 218, 117,
    159, 186, 213, 100, 172, 239, 42, 126, 130, 157, 188, 223, 122, 142, 137, 128,
    155, 182, 193, 88, 232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84,
    252, 31, 33, 99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202,
    69, 207, 74, 222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14,
    18, 54, 90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23,
    57, 75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1 ]

# Multiplication function in GF(2^8)
def multGF256(a,b):
    if (a==0) or (b==0):
        return 0
    else:
        return alog_table[(log_table[a]+log_table[b]) %255]

def random_sampling(data, num_sample):
    #print(len(data))
    #print(data.shape)
    np.random.seed(2025)
    rand_ids = np.random.choice(len(data), num_sample, replace=False)
    print(len(rand_ids))
    print('---')
    return rand_ids

def unceratainty_sampling(model, xTrain_in, num_sample):
    preds = model.predict(xTrain_in)
    pred_probs = np.min(preds, axis = 1)
    idx = 0
    #Get samples that below the mean probability
    '''
    pred_probs = []
    for sample in preds:
        #label_idx = np.where(yTrain_in[idx]==1)
        pred_prob = np.min(preds[idx])
        #idx += 1
        pred_probs.append(pred_prob)
    pred_probs = np.squeeze(np.array(pred_probs))
    '''
    #max_idxs = np.argpartition(pred_probs, -num_sample)[-num_sample:]
    min_idxs = np.argpartition(pred_probs, num_sample)[:num_sample]

    return min_idxs

parser = parse_arguments()
args = parser.parse_args()

fpath = 'AES_PTv2_D1.h5'
(X_profiling, Y_profiling), (X_attack, Y_attack) = load_ascad(fpath)

print('X_profiling: ' , X_profiling.shape)
print('Y_profiling: ' , Y_profiling.shape)
print('X_attack: ' , X_attack.shape)
print('Y_attack: ' , Y_attack.shape)
print(np.unique(Y_profiling, return_counts=False))
print(np.unique(Y_attack, return_counts=False))

save_path = 'PhaseB_{}'.format(args.name)
print(save_path)
database_folder_train = os.path.join('multi_attack_trained_models', save_path)
Path(database_folder_train).mkdir(parents=True, exist_ok=True)

#Inspect the label distribution here
epochs = args.num_epoch
validation_split = 0
batch_size = args.batch_size
#all_ids = np.load(args.all_ids)

#Load model
load_folder = args.train_folder
all_active_ids = np.load(os.path.join(load_folder, 'all_ids.npy')) #Load existing samples

all_sample_ids = np.arange(0, len(X_profiling))
non_active_ids = np.setdiff1d(all_sample_ids, all_active_ids)

xTrain_Pool, yTrain_Pool = X_profiling[non_active_ids], Y_profiling[non_active_ids]
    #print(len(xTrain_Pool[0][0]))
xTrain, yTrain = X_profiling[all_active_ids], Y_profiling[all_active_ids]

#Load model
trained_model_path = os.path.join(load_folder, 'model.keras')
model = load_model(trained_model_path)
#model = mlp_best(input_dim=len(X_profiling[0]))
model.summary()

for it in range(0, args.num_it):
    #Get sampling ids
    max_ids = unceratainty_sampling(model, xTrain_Pool, num_sample = args.num_sample)
    sampled_ids = non_active_ids[max_ids.astype(int)]
    non_active_ids = np.setdiff1d(non_active_ids, sampled_ids) #np.delete(non_active_ids, sampled_ids)
    all_active_ids = np.hstack((all_active_ids, sampled_ids))
    xTrain_Pool, yTrain_Pool = X_profiling[non_active_ids], Y_profiling[non_active_ids]
    xTrain, yTrain = X_profiling[all_active_ids], Y_profiling[all_active_ids]

    Reshaped_X_profiling = xTrain
    save_file_name = os.path.join(database_folder_train, 'model{}.keras'.format(it))
    check_file_exists(os.path.dirname(save_file_name))
    # Save model calllback
    save_model = ModelCheckpoint(save_file_name)
    callbacks=[save_model]
    y=to_categorical(yTrain, num_classes=256)

    print('Start training')
    history = model.fit(x=Reshaped_X_profiling, y=y, batch_size=batch_size, verbose = 1, validation_split=validation_split, epochs=epochs, callbacks=callbacks)
