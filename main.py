import pprint

import numpy as np
import tensorflow as tf

from data import read_data
from model2 import MemN2N
import pickle
import os.path
pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 300, "internal state dimension [300]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 2, "number of hops [7]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 10, "number of epoch to  during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 10, "clip gradients to this norm [50]")
flags.DEFINE_string("pretrain_file", "data/glove.6B.300d.txt",
                    "pre-trained glove vectors file path [../data/glove.6B.300d.txt]")
flags.DEFINE_string("train_data", "data/Restaurants_Train_v2.xml", "train gold data set path [./data/Restaurants_Train_v2.xml]")
flags.DEFINE_string("test_data", "data/Restaurants_Test_Gold.xml", "test gold data set path [./data/Restaurants_Test_Gold.xml]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS


def init_word_embeddings(word2idx):

    wt = np.random.normal(0, FLAGS.init_std, [len(word2idx), FLAGS.edim]) # wt: Contains matrix of dimension: Vocab_size * e_dim  with random values from normal distribution
    with open(FLAGS.pretrain_file, 'r') as f:
        for line in f:
            content = line.strip().split()
            if content[0] in word2idx:
                wt[word2idx[content[0]]] = np.array(list(map(float, content[1:])))
    return wt


def main(_):
    source_count = []
    source_word2idx = {}

    if os.path.isfile("abc") and os.path.isfile("def"):
        train_data = pickle.load(open("abc","rb"))
        test_data = pickle.load(open("def","rb"))
    else:
        train_data = read_data(FLAGS.train_data, source_count, source_word2idx)
        test_data = read_data(FLAGS.test_data, source_count, source_word2idx)
        pickle.dump(train_data,open("abc","wb"))
        pickle.dump(test_data,open("def","wb"))


    # test_data = read_data("./data/cust_sent.xml", source_count, source_word2idx)

   
    source_word2idx = train_data[5]

    FLAGS.pad_idx = source_word2idx['<pad>']
    FLAGS.nwords = len(source_word2idx)
    FLAGS.mem_size = train_data[4] if train_data[4] > test_data[4] else test_data[4]

    pp.pprint(flags.FLAGS.__flags)

    print('loading pre-trained word vectors...')
    FLAGS.pre_trained_context_wt = init_word_embeddings(source_word2idx)
    # pad idx has to be 0
    FLAGS.pre_trained_context_wt[FLAGS.pad_idx, :] = 0
    # FLAGS.pre_trained_target_wt = init_word_embeddings(target_word2idx)
    # for i in range(15):
    #     print " Source_data: {}, target_data: {}, Label: {}, Og_source_data: {}, og_target_data:{}".format(test_data[0][i],test_data[2][i],test_data[3][i],test_data[6][i],test_data[7][i])
    with tf.Session() as sess:
        model = MemN2N(FLAGS, sess)
        model.build_model()
        # print(np.array(train_data).shape)

        model.run(train_data, test_data)



if __name__ == '__main__':
    tf.app.run()
