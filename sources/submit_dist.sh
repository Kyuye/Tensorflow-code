job_name=wgan_dist_$(date +%Y%m%d_%H%M%S)

gcloud ml-engine jobs submit training $job_name \
--package-path=$(pwd)/sources/mintor \
--module-name=mintor.model_dist \
--staging-bucket=gs://wgan/ \
--region=us-east1 \
--scale-tier=CUSTOM \
--config=./sources/config_dist.yaml \
-- \
--on_cloud=True

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

sleep 5
gcloud ml-engine jobs stream-logs $job_name
