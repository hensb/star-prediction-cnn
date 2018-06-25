import dask.dataframe as dd
import numpy as np

from google.cloud import storage
from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten
from keras.layers import Dense, Input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 2048

def model_fn(nb_chars, embedding_matrix):
    embedding_layer = Embedding(nb_chars,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(6, activation='softmax')(x)

    model = Model(sequence_input, preds)
    return model


def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    return model


def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'text': model.inputs[0]},
                                      outputs={'sentiment': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
        builder.save()

# TODO: move this to utils.py or something? not really model related, i guess?
def get_training_data(data_source, max_nb_words, max_sequence_length, validation_split, embedding_file):
    # read texts
    df = dd.read_csv(data_source).compute().set_index("Id")
    # df = df.sample(10000)
    texts = df['Text'].tolist()

    # tokenize input
    tokenizer = Tokenizer(nb_words=max_nb_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_sequence_length)

    labels = df['Score']
    labels_index = {'test': 0}
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(validation_split * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    client = storage.Client()
    bucket = client.get_bucket('amazon-dataset')
    blob = bucket.get_blob(embedding_file)

    embeddings_index = {}
    for row in blob.download_as_string().splitlines():
        values = row.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return len(word_index) + 1, embedding_matrix, x_train, y_train, x_val, y_val