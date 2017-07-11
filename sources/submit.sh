job_name=wgan_rn_$(date +%Y%m%d_%H%M%S)

gcloud ml-engine jobs submit training $job_name \
--package-path=$(pwd)/sources/personality_replication \
--module-name=personality_replication.model \
--staging-bucket=gs://tensorflowprojects-mlengine/ \
--region=europe-west1 \
--scale-tier=BASIC_GPU 

sleep 5

gcloud ml-engine jobs stream-logs $job_name