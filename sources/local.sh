gcloud ml-engine local train \
--package-path=$(pwd)/sources/personality_replication \
--module-name=personality_replication.model \
--distributed \
--parameter-server-count=2 \
--worker-count=3 