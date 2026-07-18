# Reproducible local execution environment.
#
#   docker build -t distributed-image-pipeline .
#   docker run --rm distributed-image-pipeline --help
#
# Notes and limitations:
# - Includes a JRE so PySpark runs in local mode inside the container.
#   It does NOT talk to a Dataproc cluster; cloud runs are submitted from
#   your own authenticated environment.
# - No credentials are baked in. For GCS access, mount your own application
#   default credentials, e.g.:
#     docker run -v $HOME/.config/gcloud:/root/.config/gcloud:ro \
#       -e GOOGLE_APPLICATION_CREDENTIALS=... distributed-image-pipeline ...
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml LICENSE README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[spark,tensorflow,stats]"

ENTRYPOINT ["python", "-m", "distributed_image_pipeline.cli"]
CMD ["--help"]
