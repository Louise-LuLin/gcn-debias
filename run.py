import numpy as np
from tqdm import tqdm
import scipy
from scipy import sparse
import pickle as pkl
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import json
import random
from tqdm import tqdm
import csv
import math
import nltk
import operator
import os
from collections import defaultdict

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import networkx as nx

import easydict
import time
import sys
import tensorflow as tf
from tensorflow.python.util import deprecation

from src.minibatch import EdgeBatch, NeighborSampler
from src.model import LayerInfo, UnsupervisedSAGE


with open("G.bin", 'rb') as f:
    G = pkl.load(f)
with open("walks.bin", 'rb') as f:
    walks = pkl.load(f)
with open("X.bin", 'rb') as f:
    X = pkl.load(f)
with open("IPS.bin", 'rb') as f:
    IPS = pkl.load(f)
    
args = easydict.EasyDict({
    "infolder": "../dataset/stackoverflow/sample-218016", # yelp/sample-641938, stackoverflow/sample-218016
    "outfolder": "../dataset/stackoverflow/sample-218016/embeddings", # yelp/sample-641938/embeddings, stackoverflow/sample-218016
    "gpu": 2,
    "model": "SAGE",
    "epoch": 1,
    "batch_size": 256, # 64 for GAT; 128 for SAGE
    "dropout": 0.,
    "ffd_dropout": 0.,
    "attn_dropout": 0.,
    "vae_dropout": 0.,
    "weight_decay": 0.0,
    "learning_rate": 0.000005,
    "max_degree": 30,
    "sample1": 30,
    "sample2": 20,
    "dim1": 25,
    "dim2": 25,
    "neg_sample": 30,
    "head1": 8,
    "head2": 1
})
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
deprecation._PRINT_DEPRECATION_WARNINGS = False

def train(G, walks, X, IPS):
    # placeholders
    placeholders = {
        'batch1': tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'batch2': tf.placeholder(tf.int32, shape=(None), name='batch2'),
        'neg_sample': tf.placeholder(tf.int32, shape=(None,), name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'ffd_dropout': tf.placeholder_with_default(0., shape=(), name='ffd_dropout'),
        'attn_dropout': tf.placeholder_with_default(0., shape=(), name='attn_dropout'),
        'vae_dropout': tf.placeholder_with_default(0., shape=(), name='vae_dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }

    # batch of edges
    minibatch = EdgeBatch(G, {}, placeholders, walks, 
                          batch_size=args.batch_size, max_degree=args.max_degree, vocab_dim=0)
    # adj_info
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    # ips_info
    ips_info_ph = tf.placeholder(tf.float32, shape=IPS.shape)
    ips_info = tf.Variable(ips_info_ph, trainable=False, name="adj_info")
    # (node1, node2) -> edge_idx
    edge_idx_ph = tf.placeholder(dtype=tf.int32, shape=minibatch.edge_idx.shape)
    edge_idx = tf.Variable(edge_idx_ph, trainable=False, name='edge_idx')
    # edge_vecs
    edge_vec_ph = tf.placeholder(dtype=tf.float32, shape=minibatch.edge_vec.shape)
    edge_vec = tf.Variable(edge_vec_ph, trainable=False, name='edge_vec')

    # sample of neighbor for convolution
    sampler = NeighborSampler(adj_info)
    # two layers
    layer_infos = [LayerInfo('layer1', sampler, args.sample1, args.dim1, args.head1),
                   LayerInfo('layer2', sampler, args.sample2, args.dim2, args.head2)]

    # initialize session
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # GCN model
        model = UnsupervisedSAGE(placeholders, X, ips_info, minibatch.deg, layer_infos, args)

        sess.run(tf.global_variables_initializer(), 
                 feed_dict={adj_info_ph: minibatch.adj, 
                            ips_info_ph: IPS,
                            edge_idx_ph: minibatch.edge_idx, 
                            edge_vec_ph: minibatch.edge_vec})

        # print out model size
        para_size = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print ("Model size: {}".format(para_size))

        # begin training
        t = time.time()
        for epoch in range(args.epoch):
            minibatch.shuffle()

            iter = 0
            print ('Epoch: {} (batch={})'.format(epoch + 1, minibatch.left_edge()))
            while not minibatch.end_edge():
                # construct feed dictionary
                feed_dict, _ = minibatch.next_edgebatch_feed_dict()
                feed_dict.update({placeholders['dropout']: args.dropout})
                feed_dict.update({placeholders['ffd_dropout']: args.ffd_dropout})
                feed_dict.update({placeholders['attn_dropout']: args.attn_dropout})
                feed_dict.update({placeholders['vae_dropout']: args.vae_dropout})

                outs = sess.run([model.loss, model.mrr, model.inputs1, model.batch_size], feed_dict=feed_dict)
                if iter % 100 == 0:
                    print ('-- iter: ', '{:4d}'.format(iter),
                           'train_loss=', '{:.5f}'.format(outs[0]),
                           'train_mrr=', '{:.5f}'.format(outs[1]),
                           'time so far=', '{:.5f}'.format((time.time() - t)/60))
                iter += 1

        print ('Training {} finished!'.format(args.model))

        # save embeddings
        embeddings = []
        nodes = []
        seen = set()
        minibatch.shuffle()
        iter = 0
        while not minibatch.end_node():
            feed_dict, edges = minibatch.next_nodebatch_feed_dict()
            print ('-- iter: ', '{:4d}'.format(iter), edges)
            for p in edges:
                (n, _) = p
                if n >= len(G.nodes()):
                    print ('Gotcha!{}'.format(n))
            outs = sess.run([model.outputs1], 
                            feed_dict=feed_dict)
            # only save embeds1 because of planetoid
            for i, edge in enumerate(edges):
                node = edge[0]
                if not node in seen:
                    embeddings.append(outs[0][i, :])
                    nodes.append(node)
                    seen.add(node)
            if iter % 100 == 0:
                print ('-- iter: ', '{:4d}'.format(iter), 
                       'node_embeded=', '{}'.format(len(seen)))
            iter += 1
    return (nodes, embeddings)

# train without or with IPS
n_node = len(G.nodes())
IPS = np.ones((n_node, n_node))
(nodes, embeddings) = train(G, walks, X, IPS)
with open("embedding.bin", 'wb') as f:
    pkl.dump((nodes, embeddings), f)
    
    