# -*- coding: utf-8 -*-

import pickle
import numpy as np

import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.contrib.session_bundle import exporter
from pathlib import Path

from model.model import NERModel
from utils.train_utils import get_config_proto

tf.flags.DEFINE_string("model_dir", None, "directory to load model from")
tf.flags.DEFINE_string("export", None, "export path")
FLAGS = tf.flags.FLAGS

def main(_argv):
    """Program entry point.
    """
    if not FLAGS.model_dir:
        raise Exception("you must provide model directory")

    with tf.Session(config=get_config_proto(False)) as sess:
        # read config from pickle
        model_dir = Path(FLAGS.model_dir)
        config = pickle.load((model_dir / 'config.pkl').open('rb'))
        config.mode = "inference"

        model = NERModel(sess, config)
        model.build()

        # using best dev model
        best_dev_dir = model_dir / "best_dev"
        model.restore_model(best_dev_dir)

        export_path = FLAGS.export
        builder = saved_model_builder.SavedModelBuilder(export_path)
        prediction_input_tokens = utils.build_tensor_info(model.source_tokens)
        prediction_input_length = utils.build_tensor_info(model.source_length)
        prediction_input_segments = utils.build_tensor_info(model.segment_tokens)
        prediction_output_sequence = utils.build_tensor_info(model.viterbi_sequence)
        prediction_output_score = utils.build_tensor_info(model.viterbi_score)

        prediction_signature = signature_def_utils.build_signature_def(
            inputs={'query': prediction_input_tokens, 'query_length': prediction_input_length, "segement": prediction_input_segments},
            outputs={"sequence": prediction_output_sequence,"score":prediction_output_score},
            method_name=signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            signature_def_map={
                'predict':
                prediction_signature,
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature,
            },
            legacy_init_op=legacy_init_op,
            clear_devices=True)

        builder.save()

        print('Done exporting!')


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()