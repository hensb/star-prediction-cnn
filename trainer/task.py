import argparse
import glob
import json
import os
import time

import keras
from keras.models import load_model
import trainer.model as model
from tensorflow.python.lib.io import file_io

FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
MY_MODEL_NAME = 'my_model.hdf5'
VALIDATION_SPLIT = 0.2
EMBEDDING_FILE_GCS = 'glove.6B.100d.txt'
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 2048


def dispatch(data_file, job_dir, num_epochs):
    job_dir = create_job_dir(job_dir)
    nb_chars, embedding_matrix, x_train, y_train, x_val, y_val = \
        model.get_training_data(data_file, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, VALIDATION_SPLIT, EMBEDDING_FILE_GCS)
    my_model = model.model_fn(nb_chars, embedding_matrix)

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    checkpoint_path = FILE_PATH
    if not job_dir.startswith("gs://"):
        checkpoint_path = os.path.join(job_dir, checkpoint_path)

    # Model checkpoint callback
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')

    timestamp = str(time.time())

    # Tensorboard logs callback
    tblog = keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        write_graph=True,
        embeddings_freq=0)

    callbacks = [checkpoint, tblog]

    my_model = model.compile_model(my_model)
    my_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=128,
                 callbacks=callbacks)

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    if job_dir.startswith("gs://"):
        my_model.save(MY_MODEL_NAME)
        copy_file_to_gcs(job_dir, MY_MODEL_NAME)
    else:
        my_model.save(os.path.join(job_dir, MY_MODEL_NAME))

    # Convert the Keras model to TensorFlow SavedModel
    model.to_savedmodel(my_model, os.path.join(job_dir, 'export'))


def create_job_dir(job_dir):
    timestamp = str(time.time())
    job_dir = job_dir + "/run" + timestamp
    try:
        os.makedirs(job_dir)
    except:
        pass
    return job_dir


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file',
                        required=True,
                        type=str,
                        help='Data file local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                        required=False,
                        default='jobs',
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=5,
                        help='Maximum number of epochs on which to train')
    parse_args, unknown = parser.parse_known_args()

    dispatch(**parse_args.__dict__)
