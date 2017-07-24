python ./sources/mintor/model.py \
--on_cloud=False \
--emotion_data=/dataset/Negative.tsv \
--word_vec_map_file=/dataset/word2vec_map.json \
--log_dir=./logs/ \
--gpu_num=4 \
--task=neg_sent
