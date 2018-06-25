Predict Star Ratings from Amazon Reviews

> gcloud ml-engine jobs submit training hesc_training_job_15 --config=config.yaml --runtime-version=1.8 --module-name=trainer.task --staging-bucket=gs://hesc-ml-staging --package-path=./trainer --region=europe-west1 -- --data-file=gs://amazon-dataset/Reviews.csv