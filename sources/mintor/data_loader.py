
import pandas
import json
import os
import numpy as np
from scipy.spatial.distance import cosine

class TrainDataLoader(object):
    def __init__(self, bucket, emotion_tsv=None, train_data_csv=None, word2vec_map_json=None, on_cloud=False):
        if on_cloud:
            os.system("mkdir dataset")
            # os.system("gsutil -m cp -r gs://jejucamp2017/dataset/* $(pwd)/dataset/")
            os.system("gsutil -m cp -r gs://{}/dataset/* $(pwd)/dataset/".format(bucket))

            print("data set copy")
        
        print("reading train data...")
        if train_data_csv:
            self.train_data = self.read_datafile(os.getcwd() + train_data_csv)
        print("loading embedding map...")
        if word2vec_map_json:
            self.embedding_map = self.load_embedding_map(os.getcwd() + word2vec_map_json)
        if emotion_tsv:
            self.emo_train_data = self.read_emotions(os.getcwd() + emotion_tsv)
    

    def read_datafile(self, filename):
        data = pandas.read_csv(filename, usecols=["Sentiment", "content"])
        data = data[data["content"] != "0"]
        data["content"] = data["content"].astype("str")
        return data

    def read_emotions(self,filename):
        return pandas.read_table(filename)

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