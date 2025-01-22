gcloud ai custom-jobs create \
    --region=europe-west3 \
    --display-name=training-run \
    --config=vertex_config.yaml \
    --command 'python src/mlops/train.py' \
    --args '["--epochs", "10"]' \
