job_name=wgan_gpu_us_east_$(date +%Y%m%d_%H%M%S)

gcloud ml-engine jobs submit training $job_name \
--package-path=$(pwd)/sources/mintor \
--module-name=mintor.model_gan_test \
--staging-bucket=gs://wgan/ \
--region=europe-west1 \
--scale-tier=BASIC_GPU \
-- \
--on_cloud=True \
--bucket=jejucamp2017 \
--train_data=/dataset/twitter_emotion_v2\(p,n,N\).csv \
--word_vec_map_file=/dataset/word2vec_map.json \
--log_dir=./logs/ \
--gpu_num=1

# region list
# asia-east1
# us-east1
# us-central1
# europe-west1


# scale tier list
# BASIC: A single worker instance. This tier is suitable for learning how to use Cloud ML Engine and for experimenting with new models using small datasets.
# STANDARD_1: Many workers and a few parameter servers.
# PREMIUM_1: A large number of workers with many parameter servers.
# BASIC_GPU: A single worker instance with a GPU.
# CUSTOM: custom setting

