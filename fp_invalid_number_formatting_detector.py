import pandas
import keras

import numpy as np

from pandas import Series
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Activation
from keras.optimizers import RMSprop


chars_set = set()
max_sequence_len = 0


def get_error_df_from_xlsx(path):
    df = pandas.read_excel(path)
    index = df.index[df['False Positive #'] == 'True Positive #'][0]
    fp_df = df[:index - 1]
    tp_df = df[index + 1:].reset_index()
    return tp_df, fp_df


def convert_string_to_sequences(str_data):
    for char in str(str_data):
        chars_set.update(char)
    chars = sorted(list(chars_set))
    mapping = dict((c, i) for i, c in enumerate(chars))

    global max_sequence_len
    encoded_seq = [mapping[char]+1 for char in str(str_data)]
    if max_sequence_len < len(encoded_seq):
        max_sequence_len = len(encoded_seq)
    return encoded_seq


def transform_seq_len_to_max_size(seq):
    seq_size = len(seq)
    for _ in range(0, max_sequence_len - seq_size):
        seq.append(0)
    return seq


if __name__ == '__main__':
    # Get dataset
    tp_df, fp_df = get_error_df_from_xlsx('Invalid_Number_Formating(FP tables).xlsx')

    # Encode sequences
    tp_df['Source'] = tp_df['Source'].apply(convert_string_to_sequences)
    tp_df['Target'] = tp_df['Target'].apply(convert_string_to_sequences)
    fp_df['Source'] = fp_df['Source'].apply(convert_string_to_sequences)
    fp_df['Target'] = fp_df['Target'].apply(convert_string_to_sequences)

    tp_df['Source'] = tp_df['Source'].apply(transform_seq_len_to_max_size)
    tp_df['Target'] = tp_df['Target'].apply(transform_seq_len_to_max_size)
    fp_df['Source'] = fp_df['Source'].apply(transform_seq_len_to_max_size)
    fp_df['Target'] = fp_df['Target'].apply(transform_seq_len_to_max_size)

    # Sampling
    train_sample = fp_df.sample(frac=0.7, replace=True)
    test_sample = fp_df.drop(train_sample.index)

    fp_x_train_frame = train_sample[['Source', 'Target']]
    fp_x_test_frame = test_sample[['Source', 'Target']]
    fp_x_train_frame = fp_x_train_frame.assign(rate=np.zeros(len(fp_x_train_frame)))
    fp_x_test_frame = fp_x_test_frame.assign(rate=np.zeros(len(fp_x_test_frame)))

    train_sample = tp_df.sample(n=int(0.7*len(fp_df)), replace=True)
    test_sample = tp_df.drop(train_sample.index)

    tp_x_train_frame = train_sample[['Source', 'Target']]
    tp_x_test_frame = test_sample[['Source', 'Target']]
    tp_x_train_frame = tp_x_train_frame.assign(rate=np.ones(len(tp_x_train_frame)))
    tp_x_test_frame = tp_x_test_frame.assign(rate=np.ones(len(tp_x_test_frame)))

    x_train = pandas.concat([tp_x_train_frame, fp_x_train_frame])
    x_test = pandas.concat([tp_x_test_frame, fp_x_test_frame])

    y_train = x_train['rate']
    y_test = x_test['rate']

    x_train = x_train[['Source', 'Target']]
    x_test = x_test[['Source', 'Target']]

    x_train['concat'] = x_train['Source'] + x_train['Target']
    x_train = x_train['concat']

    x_test['concat'] = x_test['Source'] + x_test['Target']
    x_test = x_test['concat']

    x_train = np.array([keras.utils.to_categorical(x, num_classes=len(chars_set)) for x in x_train])
    x_test = np.array([keras.utils.to_categorical(x, num_classes=len(chars_set)) for x in x_test])

    num_classes = 2

    y_test_saved = y_test.copy()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('x_train_shape: ', x_train.shape)
    print('x_test_shape: ', x_test.shape)
    print('y_train shape: ', y_train.shape)
    print('y_test shape: ', y_test.shape)

    max_features = max_sequence_len * 2

    # Build model
    model = Sequential()
    model.add(LSTM(1024, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x_train, y_train)

    model_acc = model.evaluate(x_test, y_test)
    print("Accuracy: ", model_acc)
