job_name=wgan_gpu_us_east_$(date +%Y%m%d_%H%M%S)

gcloud ml-engine jobs submit training $job_name \
--package-path=$(pwd)/sources/mintor \
<<<<<<< HEAD
--module-name=mintor.model_1emo \
--staging-bucket=gs://jejucamp2017/ \
--region=us-east1 \
=======
--module-name=mintor.model_gan_test \
--staging-bucket=gs://wgan/ \
--region=europe-west1 \
>>>>>>> 8b06438a55e2847e4d0c393275b682b90d0cf127
--scale-tier=BASIC_GPU \
-- \
--on_cloud=True \
--bucket=jejucamp2017 \
--emotion_data=/dataset/Neutral.tsv \
--word_vec_map_file=/dataset/word2vec_map.json \
--log_dir=./logs/ \
--gpu_num=1

# region list
# asia-east1
# us-east1
# us-central1
# europe-west1

# --emotion_data=/dataset/Positive.tsv \
# --emotion_data=/dataset/Negative.tsv \
# --emotion_data=/dataset/Neutral.tsv \

# scale tier list
# BASIC: A single worker instance. This tier is suitable for learning how to use Cloud ML Engine and for experimenting with new models using small datasets.
# STANDARD_1: Many workers and a few parameter servers.
# PREMIUM_1: A large number of workers with many parameter servers.
# BASIC_GPU: A single worker instance with a GPU.
# CUSTOM: custom setting

