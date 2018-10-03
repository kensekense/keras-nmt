import numpy as np
import pandas as pd
from cleaner import *

from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model

#variables
num_samples = 1000 #Seems to be a number sample that doesn't give me memory issues on my computer
batch_size = 128
epochs = 500

en = []
fr = []
en_words = set()
fr_mots = set()

#open up files and load into Pandas DataFrame
with open('./text/europarl-v7.fr-en.en', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')
    for line in lines[:min(num_samples, len(lines)-1)]:
        line = cleaner(line) #clean the line
        en.append(line)

        #create vocabulary
        for word in line.split():
            if word not in en_words:
                en_words.add(word)

with open('./text/europarl-v7.fr-en.fr', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')
    for line in lines[:min(num_samples, len(lines)-1)]:
        line = cleaner(line) #clean the line
        line = 'START_ ' + line + ' _END'
        fr.append(line)

        #create vocabulary
        for mot in line.split():
            if mot not in fr_mots:
                fr_mots.add(mot)

#DATAFRAME with EN and FR side by side
language_data = pd.DataFrame(data={'en': en, 'fr':fr})

input_words = sorted(list(en_words))
target_words = sorted(list(fr_mots))
num_encoder_tokens = len(en_words)
num_decoder_tokens = len(fr_mots)

#save as csv
en_vocab = pd.DataFrame(data=input_words)
fr_vocab = pd.DataFrame(data=target_words)
en_vocab.to_csv('./frames/en-vocab.csv')
fr_vocab.to_csv('./frames/fr-vocab.csv')
print('Saved.')
print('-')

#determine the length of the max-length sentence in en and fr (by words)
en_count = []
for line in language_data['en']:
    str = line.split()
    en_count.append(len(str))
language_data['en_c'] = en_count

fr_comte = []
for line in language_data['fr']:
    str = line.split()
    fr_comte.append(len(str))
language_data['fr_c'] = fr_comte

#save as csv
language_data.to_csv('./frames/en-fr.csv')
print('Saved.')
print('-')

max_len_en = language_data['en_c'].max()
max_len_fr = language_data['fr_c'].max()

input_token_index = dict([(word, i) for i, word in enumerate(en_words)])
target_token_index = dict([(mot, i) for i, mot in enumerate(fr_mots)])

#print(max_len_en, max_len_fr)

#set up encoder and decoder input expectations
encoder_input_data = np.zeros((len(language_data['en']), max_len_en), dtype='float32')
decoder_input_data = np.zeros((len(language_data['fr']), max_len_fr), dtype='float32')
decoder_target_data = np.zeros((len(language_data['fr']), max_len_fr, num_decoder_tokens), dtype='float32')

for i, (inp, targ) in enumerate(zip(language_data['en'], language_data['fr'])):
    for t, word in enumerate(inp.split()):
        encoder_input_data[i, t] = input_token_index[word]
    for t, mot in enumerate(targ.split()):
        decoder_input_data[i,t] = target_token_index[mot]
        if t > 0:
            decoder_target_data[i, t-1, target_token_index[mot]] = 1

#model
embedding_size = 256 #latent dimensionality on keras site
encoder_inputs = Input(shape=(None,))
decoder_inputs = Input(shape=(None,))

#encoder
en_x = Embedding(num_encoder_tokens, embedding_size)(encoder_inputs)
encoder = LSTM(embedding_size, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)
encoder_states = [state_h, state_c]

#decoder
de_x = Embedding(num_decoder_tokens, embedding_size)
fde_x = de_x(decoder_inputs)
decoder_lstm = LSTM(embedding_size, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(fde_x, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

#training
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)
model.save('./models/seq2seq.h5')

#load
#from keras.models import load_model
#model = load_model('./models/seq2seq.h5')

encoder_model = Model(encoder_inputs, encoder_states)

#sampling model
decoder_state_input_h = Input(shape=(embedding_size,))
decoder_state_input_c = Input(shape=(embedding_size,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

de_x2 = de_x(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(de_x2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

reverse_input_word_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_mot_index = dict((i, mot) for mot, i in target_token_index.items())

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = target_token_index['START_']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_mot = reverse_target_mot_index[sampled_token_index]
        decoded_sentence += ' '+sampled_mot

        if(sampled_mot =='_END' or len(decoded_sentence) > max_len_fr):
            stop_condition = True

        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence

for seq_index in range(10):
    input_seq = encoder_input_data[seq_index:seq_index+1]
    decoded_sentence = decode_sequence(input_seq)
    print('Input: ', language_data['en'][seq_index])
    print('Output: ', decoded_sentence)
    print('-')
