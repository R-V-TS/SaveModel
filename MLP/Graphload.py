import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
GRAPH_PB_PATH = 'frozentensorflowModel.pb'
with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
   print(names)

   pred = [-0.536965, 0.260268, 0.381393, -0.910497, 0.62875, -0.531673, -0.860733, -0.903331, 0.499434, 0.505557, -0.9008,
        -0.99592, 0.405563, 0.33531, -0.95923, -0.998755]
   pred = np.reshape(pred, (1, 16))
   v = sess.run("dense_output/BiasAdd:0", feed_dict={"dense_input/Tanh:0": pred})
   print(v)
