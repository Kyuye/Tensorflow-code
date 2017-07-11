gcloud ml-engine jobs submit training wgan_rn_$(date +%Y%m%d_%H%M%S) \
--package-path=$(pwd)/sources/personality_replication \
--module-name=personality_replication.model \
--staging-bucket=gs://tensorflowprojects-mlengine/ \
--region=europe-west1 \
--scale-tier=BASIC_GPU 