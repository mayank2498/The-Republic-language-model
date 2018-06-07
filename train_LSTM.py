from cleaned_data import get_training_data
from keras.preprocessing.text import Tokenizer
from numpy import array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

file_name = 'republic_sequences.txt'
data = load_doc(file_name)
lines = data.split('\n')
lines = lines[:10000]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)


# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# separate training and test set
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]


# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=100)
 
# save the model to file
model.save('model.h5')
# save the tokenizer
from pickle import dump
dump(tokenizer, open('tokenizer.pkl', 'wb'))