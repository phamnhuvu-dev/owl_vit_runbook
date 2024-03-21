import apache_beam as beam
import tensorflow_datasets as tfds
from scenic.projects.baselines.clip import download
from skimage import data


# Download skimage data
data.download_all()

# Download COCO dataset
builder = tfds.builder("coco/2017")
flags = ["--direct_num_workers=4", "--direct_running_mode=multi_processing"]
builder.download_and_prepare(
    download_config=tfds.download.DownloadConfig(
        beam_runner="DirectRunner",
        beam_options=beam.options.pipeline_options.PipelineOptions(flags=flags),
    )
)