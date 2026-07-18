# Data card — TensorFlow Flowers dataset

## Source

- Canonical archive: `http://download.tensorflow.org/example_images/flower_photos.tgz`
  (published by the TensorFlow team; also available as the `tf_flowers`
  dataset in TensorFlow Datasets).
- A copy preprocessed into public GCS buckets (`gs://flowers-public/*/*.jpg`)
  was used in the original coursework via Google's "Fast and Lean Data
  Science" course material. Availability of that public bucket is not
  guaranteed; prefer the canonical archive.

## Size and classes

- **3,670 JPEG images** (per the TensorFlow Datasets catalog), one label per
  image, labelled by parent directory.
- **Five classes**: `daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`.
- Class distribution is uneven. Commonly reported per-class counts are
  daisy 633, dandelion 898, roses 641, sunflowers 699, tulips 799 (these sum
  to 3,670); verify against your downloaded copy with
  `find flower_photos -name '*.jpg' | awk -F/ '{print $(NF-1)}' | sort | uniq -c`.

## Licence

The images were collected from Flickr and are individually licensed,
predominantly under Creative Commons Attribution licences. The downloaded
archive contains a `LICENSE.txt` with per-image attribution and licence
details — consult it before redistributing images. This repository does not
redistribute the images.

## Intended use (in this project)

- Benchmarking preprocessing throughput (local vs Spark, partition counts,
  raw JPEG vs TFRecord input) and downstream training *throughput*.
- Small-scale, reproducible experiments on a single machine or a small
  Dataproc cluster.

## Out-of-scope use

- State-of-the-art flower classification: the dataset is small and the model
  in this project is deliberately tiny.
- Any conclusion about scaling behaviour at data volumes orders of magnitude
  larger than ~3,700 images.
- Production botany/plant-identification applications.

## Known limitations

- Small dataset: throughput benchmarks include fixed overheads (Spark job
  startup, TF graph tracing) that would amortise differently at scale.
- Uneven class balance; splits in this project are stratified per class.
- Variable image resolutions and photographic conditions; a handful of files
  may be non-JPEG or corrupted — the pipeline counts and reports such files
  rather than failing silently.
