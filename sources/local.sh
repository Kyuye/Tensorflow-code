gcloud ml-engine local train \
--package-path=$(pwd)/sources/personality_replication \
--module-name=personality_replication.model_single \
# --distributed \
# --parameter-server-count=5 \
# --worker-count=10