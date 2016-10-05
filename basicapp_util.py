""" basicapp_util.py - Utility for basicapp.py """

import tensorflow as tf
import numpy as np

log_dir = 'log'


def write_graph():
    # Save the current default tf graph
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    tf.train.SummaryWriter(log_dir, tf.get_default_graph())


def gen_DAG(n=None, m=None, incap=3, lb=-10, ub=10, verbose=False):
    """ Random generate a DAG, the raw inputs and the outputs
    :param n: number of vertices
    :param m: number of edges
    :param incap: maximum input degree
    :param lb, ub: lower and upper bounds for input
    :param verbose:
    :return vertices: (n, 2) narray as the input to all the vertices
    :return edges: m ordered paires represent the directed edges
    :return output: (n, 2) narray as the ground truth output of all the vertices
    """

    # Vertices
    if n is None:
        n = np.random.randint(1, 50)
    vertices = np.random.randint(lb, ub, (n, 2))

    # Edges
    if n >= incap:
        edge_ub = (n - incap) * incap + incap * (incap - 1) / 2
    else:
        edge_ub = n * (n - 1) / 2

    if m is None:
        m = np.random.randint(0, edge_ub + 1)

    if m > edge_ub:
        raise RuntimeError('Too many edges')

    dgr = np.zeros(n, dtype=np.int)  # input degrees
    g = np.zeros((n, n), dtype=np.int)  # connected edges

    edges = []
    for i in xrange(0, m):
        while True:
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u > v:
                u, v = v, u
            if not(u == v or dgr[v] == incap or g[u][v] == 1):
                edges.append((u, v))
                g[u][v] = 1
                dgr[v] += 1
                break

    # Outputs
    outputs = np.zeros((n, 2))
    for i in xrange(0, n):
        outputs[i] = vertices[i]
        for j in xrange(0, i):
            if g[j][i] == 1:
                outputs[i] += outputs[j]
        outputs[i] /= 2
        outputs[i][0], outputs[i][1] = outputs[i][1], outputs[i][0]

    if verbose:
        print n, m, incap
        print edges
        print vertices
        print outputs
    return vertices, edges, outputs
