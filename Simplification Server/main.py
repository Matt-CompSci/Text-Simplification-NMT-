import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io
import time

# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
                 if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  #w = re.sub(r"([?.!,¿])", r" \1 ", w)
  #w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w

# Loads data from a file
def load_data(path):
  input_file = os.path.join(path)

  # Open the file read-only
  with open(input_file, "r") as f:
    # Convert file contents to ascii
    data = unicode_to_ascii(f.read())
  return data.split('\n')

# Load sentence pairs
normal_sentences = load_data("normal.training.txt")
simple_sentences = load_data("simple.training.txt")

# Preprocess sentences pairs
normal_sentences = [preprocess_sentence(sentence) for sentence in normal_sentences if sentence != ""]
simple_sentences = [preprocess_sentence(sentence) for sentence in simple_sentences if sentence != ""]

sentence_amount = 15000
normal_sentences = normal_sentences[:sentence_amount]
simple_sentences = simple_sentences[:sentence_amount]

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token="OOV")
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
  return tensor, lang_tokenizer

def load_dataset(targ_lang, inp_lang, num_examples=None):
  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


num_examples = 15000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(normal_sentences, simple_sentences, num_examples)

words = len(inp_lang.word_index)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 90-10 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.1)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 300
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
      super(Encoder, self).__init__()
      self.batch_sz = batch_sz
      self.enc_units = enc_units
      self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
      self.gru = tf.keras.layers.GRU(self.enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
      x = self.embedding(x)
      output, state = self.gru(x, initial_state=hidden)
      return output, state

    def initialize_hidden_state(self):
      return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
      super(BahdanauAttention, self).__init__()
      self.W1 = tf.keras.layers.Dense(units)
      self.W2 = tf.keras.layers.Dense(units)
      self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
      # query hidden state shape == (batch_size, hidden size)
      # query_with_time_axis shape == (batch_size, 1, hidden size)
      # values shape == (batch_size, max_len, hidden size)
      # we are doing this to broadcast addition along the time axis to calculate the score
      query_with_time_axis = tf.expand_dims(query, 1)

      # score shape == (batch_size, max_length, 1)
      # we get 1 at the last axis because we are applying score to self.V
      # the shape of the tensor before applying self.V is (batch_size, max_length, units)
      score = self.V(tf.nn.tanh(
          self.W1(query_with_time_axis) + self.W2(values)))

      # attention_weights shape == (batch_size, max_length, 1)
      attention_weights = tf.nn.softmax(score, axis=1)

      # context_vector shape after sum == (batch_size, hidden_size)
      context_vector = attention_weights * values
      context_vector = tf.reduce_sum(context_vector, axis=1)

      return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
      super(Decoder, self).__init__()
      self.batch_sz = batch_sz
      self.dec_units = dec_units
      self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
      self.gru = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
      self.fc = tf.keras.layers.Dense(vocab_size)

      # used for attention
      self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
      # enc_output shape == (batch_size, max_length, hidden_size)
      context_vector, attention_weights = self.attention(hidden, enc_output)

      # x shape after passing through embedding == (batch_size, 1, embedding_dim)
      x = self.embedding(x)

      # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
      x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

      # passing the concatenated vector to the GRU
      output, state = self.gru(x)

      # output shape == (batch_size * 1, hidden_size)
      output = tf.reshape(output, (-1, output.shape[2]))

      # output shape == (batch_size, vocab)
      x = self.fc(output)

      return x, state, attention_weights


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units)", attention_result.shape)
print("Attention weights shape: (batch_size, sequence_length, 1)", attention_weights.shape)

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)
print('Decoder output shape: (batch_size, vocab size)', sample_decoder_output.shape)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                  decoder=decoder)

checkpoint.restore("./AttentionModel/ckpt-5").expect_partial()

def evaluate(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = preprocess_sentence(sentence)

  temp_sentence = [word for word in sentence.split(" ") if word in inp_lang.word_index.keys()]
  inputs = [inp_lang.word_index[i] for i in temp_sentence]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot

def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)

  #print('Input:', sentence)
  #print('Predicted translation:', result)

  words = result.split(" ")
  idxToRemove = []
  for idx, word in enumerate(words):
    if idx != 0:
      if word == words[idx - 1]:
        idxToRemove.append(idx)
  for idx in reversed(idxToRemove):
    del words[idx]
  return " ".join(words)

from urllib.parse import parse_qs, urlparse
import http
import json
import socketserver
from http.server import SimpleHTTPRequestHandler
PORT = 7777

class Server(SimpleHTTPRequestHandler):
    def do_GET(self):
        text_to_simplify = parse_qs(urlparse(self.path).query).get("text", None)
        print("Input:", text_to_simplify)
        output = ""
        for sentence in text_to_simplify[0].split("."):
          tempSentence = sentence + "."
          tempSentence = preprocess_sentence(tempSentence)
          output += translate(tempSentence)
        print("Output:", output)

        output_json = json.dumps({
            "output": output
        }, indent=4)
        print(output_json)

        self.send_response(200, "OK")
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(output_json, "UTF-8"))

    def serve_forever(port):
        socketserver.TCPServer(("", port), Server).serve_forever()

print("Server loaded on PORT:", PORT)
Server.serve_forever(PORT)

# # 'He', 'became', 'Prime', 'Minister', 'two', 'days', 'before', 'the', 'Great', 'Depression', 'began', '.'
# # He became Prime Minister two days before the great depression began