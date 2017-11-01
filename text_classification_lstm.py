import pandas
import keras

import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM

MAX_DOCUMENT_LENGTH = 50
EMBEDDING_SIZE = 200
n_words = 0
MAX_LABEL = 15
WORDS_FEATURE = 'words'  # Name of the input words feature.


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare training and testing data
    dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia')
    x_train = pandas.Series(dbpedia.train.data[:, 1])
    y_train = pandas.Series(dbpedia.train.target)
    x_test = pandas.Series(dbpedia.test.data[:, 1])
    y_test = pandas.Series(dbpedia.test.target)

    # создаем единый словарь (слово -> число) для преобразования
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)

    # Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
    x_train_sequences = tokenizer.texts_to_sequences(x_train)
    x_test_sequences = tokenizer.texts_to_sequences(x_test)

    print(u'Преобразуем описания заявок в векторы чисел...')
    tokenizer = Tokenizer(num_words=MAX_DOCUMENT_LENGTH)
    X_train = tokenizer.sequences_to_matrix(x_train_sequences, mode='binary')
    X_test = tokenizer.sequences_to_matrix(x_test_sequences, mode='binary')
    print('Размерность X_train:', X_train.shape)
    print('Размерность X_test:', X_test.shape)

    num_classes = len(y_test)

    y_test_saved = y_test.copy()
    print(u'Преобразуем категории в матрицу двоичных чисел '
          u'(для использования categorical_crossentropy)')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    # максимальное количество слов для анализа
    max_features = 1000

    print(u'Собираем модель...')
    model = Sequential()
    model.add(Embedding(max_features, 200))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(len(y_test), activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    print(u'Тренируем модель...')
    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=2,
                        validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test,
                           batch_size=32, verbose=1)
    print(u'Оценка теста: {}'.format(score[0]))
    print(u'Оценка точности модели: {}'.format(score[1]))

    predictions = model.predict(X_test)
