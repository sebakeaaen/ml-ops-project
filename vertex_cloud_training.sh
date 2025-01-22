gcloud ai custom-jobs create \
    --region=europe-west3 \
    --display-name=test-run \
    --config=vertex_config.yaml \
    --command 'dvc pull' \
    --command 'python src/mlops/train.py' \
    --args '["--epochs", "10"]'
