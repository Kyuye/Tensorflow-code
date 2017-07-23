
import numpy as np
import tensorflow as tf
import math

class Preprocessor(object):
    def __init__(self, embedding_map, batch_size, max_document_length):
        self.embedding_map = embedding_map
        self.max_document_length = max_document_length
        self.max_object_pairs_num = self.nCr(max_document_length,2)-1
        self.batch_size = batch_size

    def nCr(self, n,r=2):
        f = math.factorial
        return f(n) / f(r) / f(n-r)

    def embedding_padding(self, data):
        seq_vec = []
        for seq in data:
            embeds = np.array([self.embedding_map[i] if i in self.embedding_map else self.embedding_map["UNK"] for i in seq.split(' ')])
            seq_vec.append(np.pad(embeds, ((0, 150-embeds.shape[0]), (0,0)), 'constant'))
        return seq_vec

    def get_batch(self, data, iterator):
        fullbatch_size = len(data)
        start = (iterator*self.batch_size)%fullbatch_size
        end = ((iterator+1)*self.batch_size)%fullbatch_size
        if start > end:
            start = fullbatch_size - self.batch_size
            end = fullbatch_size
        train_batch = data[start:end]
        
        seq_vec = []
        for seq in train_batch['content']:
            embeds = np.array([self.embedding_map[i] if i in self.embedding_map else self.embedding_map["UNK"] for i in seq.split(' ')])
            seq_vec.append(np.pad(embeds, ((0, 150-embeds.shape[0]), (0,0)), 'constant'))

        indices = []
        for s in train_batch['Sentiment']:
            if s == "Neg":
                indices.append(0)
            elif s == "neutral":
                indices.append(1)
            elif s == "Pos":
                indices.append(2)
            else:
                indices.append(1)
        
        return seq_vec, indices


    def pairing(self, embeds):
        pair_set = []
        for seq in tf.unstack(embeds):
            comb = list(range(self.max_document_length))
            
            ids = []
            while len(comb) > 2:
                ids += list(map(lambda x: [comb[0], x], comb))[1:]
                del comb[0]
                
            pair = tf.nn.embedding_lookup(seq, ids)
            pair_set.append(tf.concat([pair[:,0], pair[:,1]], axis=1))

        return tf.stack(pair_set)
    