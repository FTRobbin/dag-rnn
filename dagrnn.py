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
    """
    :param composers: an ordered list of callables that define the composer functions for nodes of different
                      input degrees. These functions are expected to take batch_size as the first dimension
    :param core: a callable that defines the core function. This function are expected to take batch_size as
                 the first dimentsion. May support tf.nn.rnn_cell later
    :return output: a tensor represents the output of the DAGRNN, its shape should be [#output_nodes, ...]
    """
    return tf.constant(0)

def analyze(vertices, edges):
    """
    :param vertices: a numpy array of shape [#node, ...] is served as inputs for the nodes
    :param edges: a list of #edges pairs that represents the edges in the DAG
    :return: a dict that is supposed to feed into a DAGRNN defined by assemble to compute the DAG
    """
    return {}
