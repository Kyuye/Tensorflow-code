
import pandas
import json
import os
import numpy as np
from scipy.spatial.distance import cosine

class TrainDataLoader(object):
    def __init__(self, train_data_csv, word2vec_map_json, on_cloud=False):
        if on_cloud:
            os.system("mkdir dataset")
            os.system("gsutil -m cp -r gs://wgan/dataset/* $(pwd)/dataset/")
            print("data set copy")
        
        print("reading train data...")
        self.train_data = self.read_datafile(os.getcwd() + train_data_csv)
        print("loading embedding map...")
        self.embedding_map = self.load_embedding_map(os.getcwd() + word2vec_map_json)
    

    def read_datafile(self, filename):
        data = pandas.read_csv(filename, usecols=["Sentiment", "content"])
        data = data[data["content"] != "0"]
        data["content"] = data["content"].astype("str")
        return data


    def load_embedding_map(self, mapfile):
        with open(mapfile) as data_file:    
            data = json.load(data_file)
        return data

    def vec2word(self, embed):
        sim_word = ("", 1)
        for v in self.embedding_map:
            cos_dist = cosine(embed, np.array(self.embedding_map[v]))
            if cos_dist < sim_word[1]:
                sim_word = (v, cos_dist)
        return sim_word[0]