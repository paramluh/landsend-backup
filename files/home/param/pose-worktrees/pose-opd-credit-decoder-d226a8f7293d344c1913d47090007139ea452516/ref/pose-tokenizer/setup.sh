sudo apt update && sudo apt install -y curl ffmpeg tmux python3-jwt python3-cryptography

python3 scripts/whitelist_db_ip.py

curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

gcloud auth activate-service-account --key-file=service.json

curl -LsSf https://astral.sh/uv/install.sh | sh


source $HOME/.local/bin/env

ln -sfn /mnt/somfs/pose_conditioning pose_files

ls pose_files/ | grep '_poses\.npz$' | sort > manifest.txt
echo "manifest.txt: $(wc -l < manifest.txt) clips"

uv sync

echo "Downloading test_set/ from gs://data-sharing-xabi/pose-test-set/ ..."
mkdir -p test_set
gsutil -m -q cp -r "gs://data-sharing-xabi/pose-test-set/*" test_set/