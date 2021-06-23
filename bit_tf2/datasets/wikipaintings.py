"""Wikipaintings datasets."""
import os.path

import dataclasses
import yaml

import tensorflow_datasets as tfds
import tensorflow as tf

# TODO(wikipaintings): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(wikipaintings): BibTeX citation
_CITATION = """
"""

_URL = "https://github.com/chrstn-hntschl/datasets/tree/master/wikipaintings"
_DATASET_URL = "https://my.hidrive.com/api/sharelink/download?id=ErTgRlMv"


@dataclasses.dataclass
class WikipaintingsConfig(tfds.core.BuilderConfig):
    split: int = 100


class Wikipaintings(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Wikipaintings_100 dataset."""

    VERSION = tfds.core.Version('1.0.0')

    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        # `name` (and optionally `description`) are required for each config
        WikipaintingsConfig(name="Wikipaintings_5", description='', split=5),
        WikipaintingsConfig(name="Wikipaintings_10", description='', split=10),
        WikipaintingsConfig(name="Wikipaintings_20", description='', split=20),
        WikipaintingsConfig(name="Wikipaintings_40", description='', split=40),
        WikipaintingsConfig(name="Wikipaintings_60", description='', split=60),
        WikipaintingsConfig(name="Wikipaintings_80", description='', split=80),
        WikipaintingsConfig(name="Wikipaintings_100", description='', split=100),
    ]
    # pytype: enable=wrong-keyword-args

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "category": tfds.features.ClassLabel(num_classes=22),
            }),
            supervised_keys=("image", "category"),
            homepage=_URL,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        wikipaintings_path = dl_manager.download_and_extract(_DATASET_URL)
        descriptor_path = os.path.join(wikipaintings_path, f"Wikipaintings_{self.builder_config.split}.yaml")

        with tf.io.gfile.GFile(descriptor_path) as descr_f:
            descriptor = yaml.load(descr_f)

        self.info.features["category"].names = descriptor["categories"]
        if os.path.isabs(descriptor["basepath"]):
            raise AssertionError(f"Absolute image basepath is not supported! "
                                 f"Images must be located within descriptor directory!")
        images_dir_path = os.path.join(wikipaintings_path, descriptor["basepath"])

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "images_dir_path": images_dir_path,
                    "subset": "train",
                    "descriptor": descriptor
                }),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "images_dir_path": images_dir_path,
                    "subset": "val",
                    "descriptor": descriptor
                }),
        ]

    def _generate_examples(self, images_dir_path, subset, descriptor):
        """Generates images and labels given the image directory path.

        Args:
          images_dir_path: path to the directory where the images are stored.
          subset:
          descriptor:

        Yields:
          The image path, and its corresponding label and filename.

        """
        for category in self.info.features["category"].names:
            for idx in descriptor[subset]["gt"][category]:
                yield idx, {
                    "image": os.path.join(images_dir_path, descriptor[subset]["images"][idx]),
                    "category": category
                }
