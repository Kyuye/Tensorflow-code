gcloud ml-engine local train \
--package-path=$(pwd)/sources/personality_replication \
--module-name=personality_replication.model_dist \
--distributed \
--parameter-server-count=3 \
--worker-count=2