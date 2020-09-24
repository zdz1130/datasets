"""DRIVE dataset."""

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import tensorflow_io as tfio
import os

# TODO(DRIVE): BibTeX citation
_CITATION = """
link to grand-challenge webpage: https://drive.grand-challenge.org/
"""

# TODO(DRIVE):
_DESCRIPTION = """
  The photographs for the DRIVE database were obtained from a diabetic 
  retinopathy screening program in The Netherlands. The screening population 
  consisted of 400 diabetic subjects between 25-90 years of age. Forty
  photographs have been randomly selected, 33 do not show any sign of diabetic 
  retinopathy and 7 show signs of mild early diabetic retinopathy.
  The dataset has been separated to test and traning sets, each contains 20
  images. For the training images, a single manual segmentation of the 
  vasculature is available. For the test cases, two manual segmentations are 
  available; one is used as gold standard, the other one can be used to compare 
  computer generated segmentations with those of an independent human observer. 
  Furthermore, a mask image is available for every retinal image, indicating 
  the region of interest. All human observers that manually segmented the 
  vasculature were instructed and trained by an experienced ophthalmologist.
"""


class Drive(tfds.core.GeneratorBasedBuilder):
  """TODO(DRIVE): Short description of my dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  Please set manual_dir as 'gs://bme590/dingzhe_zheng/datasets.zip'
  """

  # TODO(DRIVE): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # TODO(DRIVE): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(),
            'mask': tfds.features.Image(),
            'manual': tfds.features.Image()
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        # Homepage of the dataset for documentation
        homepage='https://drive.grand-challenge.org/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(DRIVE): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    data_dir = dl_manager.manual_dir
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "image": os.path.join(data_dir,'DRIVE','training','images'), 
                "mask": os.path.join(data_dir,'DRIVE','training','mask'),
                "manual": os.path.join(data_dir,'DRIVE','training','1st_manual')
            }
        )
    ]

  def _generate_examples(self, image, mask, manual):
    """Yields examples."""
    image_names = tf.io.gfile.listdir(image)
    mask_names = tf.io.gfile.listdir(mask)
    manual_names = tf.io.gfile.listdir(manual)
    
    for i in range(len(image_names)):
        image = tf.io.read_file(os.path.join(image, image_names[i]))
        mask = tf.io.read_file(os.path.join(mask, mask_names[i]))
        manual = tf.io.read_file(os.path.join(manual, manual_names[i]))
        decoded_image = tfio.experimental.image.decode_tiff(image)
        decoded_mask = tfio.experimental.image.decode_tiff(mask)
        decoded_manual = tfio.experimental.image.decode_tiff(manual)
        yield file, {
            'image': decoded_image,
            'mask': decoded_mask,
            'manual': decoded_manual
        }

