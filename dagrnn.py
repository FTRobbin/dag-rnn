""" dagrnn.py - Supporting Library for creating DAG recursive models in Tensorflow

DAG recursive models can utilize the prior knowledge of structured data. And this library is built to
easy the creation of such models and provide a performance boost through batched ops.

Here is an overview of functions available in this library.

* Backbone interfaces.
    - assemble: assemble composer functions and core functions to build the main while loop
    - analyze: analyze the given DAG graph and provide input for the execution
"""

import tensorflow as tf

def assemble(composers, core):
    return tf.constant(0)

def analyze(vertices, edges):
    return {}
