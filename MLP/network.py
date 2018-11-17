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

def create_network():
    model = models.Sequential()
    model.add(layers.Dense(16, activation="tanh", input_dim=16, name="dense_input"))
    model.add(layers.Dense(8, activation="tanh"))
    model.add(layers.Dense(4, activation="tanh"))
    model.add(layers.Dense(1, activation="linear", name="dense_output"))
    model.compile(metrics=[mse, rmse, r_pow], optimizer="rmsprop", loss="mse")
    return model