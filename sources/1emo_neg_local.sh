python ./sources/mintor/model_1emo_local.py \
--on_cloud=False \
--emotion_data=/dataset/Negative.tsv \
--word_vec_map_file=/dataset/word2vec_map.json \
--log_dir=./logs/ \
--gpu_num=4
