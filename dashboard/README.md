# Streamlit benchmark dashboard

The recruiter-facing dashboard reads the committed measured benchmark CSVs under `reports/tables/` and computes all headline metrics dynamically.

## Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Deploy on Streamlit Community Cloud

1. Connect this GitHub repository to Streamlit Community Cloud.
2. Select the deployment branch (normally `main`).
3. Set the main file path to `dashboard/app.py`.
4. Deploy.

No API keys, cloud credentials, raw image dataset, TensorFlow installation, or Spark runtime are required to render the dashboard. The app only reads committed benchmark result CSVs.

## Dashboard sections

- Overview: headline measured KPIs and key findings.
- Preprocessing: local vs local-mode PySpark runtime and throughput across partition counts.
- JPEG vs TFRecord: standalone input-pipeline throughput and iteration-time comparisons.
- Training: TensorFlow epoch time, throughput, validation accuracy and loss.
- Experiment Explorer: raw measured CSVs with downloads.
- Methodology: experiment design, controls and limitations.

The current Spark results measure partition scaling on one GitHub Actions runner. They are not presented as multi-worker Dataproc scaling results.
