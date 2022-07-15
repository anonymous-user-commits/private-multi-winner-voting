# @title Copyright 2020 TensorFlow Constrained Optimization Authors. Double-Click Here for License Information.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# @title Import Modules
import os
import sys
import tempfile
import urllib

import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import numpy as np

import tensorflow_constrained_optimization as tfco

from tensorflow_metadata.proto.v0 import schema_pb2
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tf_example_record

# @title Fairness Indicators related imports
import tensorflow_model_analysis as tfma
import fairness_indicators as fi
from google.protobuf import text_format
import apache_beam as beam


#@title Enable Eager Execution and Print Versions
if tf.__version__ < "2.0.0":
  tf.compat.v1.enable_eager_execution()
  print("Eager execution enabled.")
else:
  print("Eager execution enabled by default.")

print("TensorFlow " + tf.__version__)
print("TFMA " + tfma.VERSION_STRING)
print("TFDS " + tfds.version.__version__)
print("FI " + fi.version.__version__)

celeb_a_builder = tfds.builder("celeb_a", data_dir=gcs_base_dir, version='2.0.0')

celeb_a_builder.download_and_prepare()

num_test_shards_dict = {'0.3.0': 4, '2.0.0': 2} # Used because we download the test dataset separately
version = str(celeb_a_builder.info.version)
print('Celeb_A dataset version: %s' % version)