Distributed Image Processing Pipeline (Spark + TensorFlow on GCP)

This project implements a distributed data pipeline that converts large image datasets into TFRecord format using PySpark and TensorFlow on Google Cloud Dataproc. It demonstrates scalable preprocessing, parallel file generation, and throughput benchmarking for machine learning data pipelines.

Originally developed as Big Data coursework, the repository has been refactored into a reproducible, portfolio-ready project that highlights distributed computing and data engineering practices.

Key features

Distributed image preprocessing pipeline using PySpark RDDs

Parallel TFRecord generation optimised for downstream ML workloads

Google Cloud Storage integration for scalable data access

Benchmarking framework comparing raw image reads vs TFRecord reads

Modular project structure separating reusable code from notebooks

Tech stack

Python

PySpark

TensorFlow

Google Cloud Platform (Dataproc + GCS)

Jupyter Notebook

Setup

Create and activate a virtual environment

Install dependencies

pip install -r requirements.txt

(Optional) Configure environment variables for GCP execution

export GCP_PROJECT_ID=your-project-id
export GCS_BUCKET_URI=gs://your-bucket
Running the pipeline

The Spark TFRecord writer can be executed as a standalone module:

python src/spark_tfrecord_writer.py

The full experimental workflow and benchmarking analysis are available in the notebooks directory.

Repository structure
.
├── data/
│   ├── processed/       # Generated datasets and intermediate outputs
│   └── raw/             # Source datasets
├── models/              # Model artifacts and experiment outputs
├── notebooks/           # Exploratory analysis and experiments
├── reports/             # Figures and report assets
├── scripts/             # Utility and orchestration scripts
└── src/                 # Reusable pipeline modules
Notes

The original coursework notebook is preserved for reference

A sanitized public notebook is included for reproducibility

The focus of this project is scalable data preprocessing rather than model training
