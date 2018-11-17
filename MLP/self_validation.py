import my_func.mapminmax as mapminmax
from MLP.read_dataMLP import read_data
from MLP.network import create_network
import tools.model_saver as ms
import numpy as np
import keras.backend as K
import tensorflow as tf


# Read Data
#metrics = ["PSNR", "PSNRHVSM", "PSNRHVS", "PSNRHMA", "PSNRHA", "FSIM", "SSIM", "MSSSIM", "GMSD", "SRSIM", "HaarPSI",
#               "VSI", "MAD_index", "DSI", "RFSIM", "GSM", "IWSSIM", "IWPSNR", "WSNR", "SFF", "IFC", "VIF", "NQM", "ADM",
#               "IGM", "PSIM", "ADDSSIM", "ADDGSIM", "DSS", "CVSSI"]
#    filters = ["DCTF", "BM3D"]
#    dbs = ["tampere17", "test"]
#    hows = ["divided_latitude", "stripped_latitude", "rated_latitude", "sparsed_latitude", "divided_longitude",
#            "stripped_longitude", "rated_longitude", "sparsed_longitude"]

(Load_data, Load_label) = read_data(metric=0) ## standart:  metric = 0, filter = 1, validation = self

#Normalize Data -1 to 1
for i in range(Load_data.shape[1]):
    (Load_data[:, i], x, y) = mapminmax.mapminmax(Load_data[:, i])

#Normalize label -1 to 1
# xmax - max in array
# xmin - min in array
(Load_label, xmax, xmin) = mapminmax.mapminmax(Load_label)
print("xmax = ", xmax, " xmin = ", xmin)
print("max_x = ", np.amax(Load_label))


# Division data
# Train_data = 70%   Test data = 15%   Validation data = 15%
length = Load_data.shape[0]
traning_max_position = int((length * 0.7) - 1)
test_max_position = int((length * 0.15) + traning_max_position)
validating_max_position = int((length*0.15) + test_max_position)

#Train_data = Load_data[0:traning_max_position, :]
#Test_data = Load_data[traning_max_position+1:test_max_position, :]
#Validation_data = Load_data[test_max_position+1:validating_max_position, :]

#Train_label = Load_label[0:traning_max_position, :]
#Test_label = Load_label[traning_max_position+1:test_max_position, :]
#Validation_label = Load_label[test_max_position+1:validating_max_position, :]

from sklearn.model_selection import train_test_split
(Train_data, Test_data, Train_label, Test_label) = train_test_split(Load_data, Load_label, test_size=0.1)

print(Train_data.shape)
## Network
import keras


from tensorflow.python.tools import freeze_graph
from keras import layers, models, backend, initializers

model_batch_size = 10


def mse(y_true, y_pred):
    return (backend.sum(backend.square(y_true - y_pred))/model_batch_size)

def rmse(y_true, y_pred):
    return backend.sqrt(backend.sum(backend.square(y_true - y_pred))/model_batch_size)

def r_pow(y_true, y_pred):
   SS_res = (backend.sum(backend.square(y_true - y_pred)))
   SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
   return (1 - (SS_res/SS_tot))

with tf.Graph().as_default():
    with tf.Session() as sess:
        model = models.Sequential()
        model.add(layers.Dense(16, activation="tanh", input_dim=16, name="dense_input"))
        model.add(layers.Dense(8, activation="tanh"))
        model.add(layers.Dense(4, activation="tanh"))
        model.add(layers.Dense(1, activation="linear", name="dense_output"))
        model.compile(metrics=[mse, rmse, r_pow], optimizer="rmsprop", loss="mse")

        model.fit(Train_data, Train_label, batch_size=10, epochs=30, verbose=2)

        # Initialize all variables
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.save(sess, './tensorflowModel.ckpt')
        tf.train.write_graph(sess.graph.as_graph_def(), '.', 'tensorflowModel.pbtxt', as_text=True)

freeze_graph.freeze_graph('tensorflowModel.pbtxt', "", False,
                          './tensorflowModel.ckpt', "dense_output/BiasAdd",
                           "save/restore_all", "save/Const:0",
                           'frozentensorflowModel.pb', True, ""
                         )
