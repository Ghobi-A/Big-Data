# Original MSc coursework (historical archive)

This directory preserves the original INM432 Big Data coursework (2023) exactly as
submitted, for historical reference. **Nothing in this directory is maintained or
intended to be run.** The modern, reproducible implementation lives in
`src/distributed_image_pipeline/`.

Contents:

- `Big_Data_CW_Final.ipynb` — the original coursework notebook (Colab-based,
  contains coursework instructions and the original student GCP project ID).
- `big-data-cw-final.pdf` — PDF export of the submitted coursework.
- `spark_tfrecord_writer.py` — the first extraction of reusable code from the
  notebook. Superseded by the `distributed_image_pipeline` package, which fixes
  its hardcoded GCP configuration, silent unknown-label handling, and 2%
  default sampling.
- `requirements.txt` — the original dependency file (incomplete: it omitted
  `pyspark` and `tensorflow`). Superseded by `pyproject.toml`.

A sanitised copy of the notebook (personal GCP project ID removed) is kept in
`notebooks/archive/Big_Data_CW_Final_public.ipynb`.
