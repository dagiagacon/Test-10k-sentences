import tensorflow as tf
import data as DataHelper

samples, labels = DataHelper.get_train_set()
test_samples, test_labels = DataHelper.get_test_set()
vocab_src, vocab_tgt = DataHelper.get_vocabs()
batch_size = 64
hidden_size = 100


class LSTMcell():
    def __init__(self, hidden_size, input_size, batch_size):
        """
        :param hidden_size: dimension of hidden unit
        :param input_size: dimension of word2vec
        :param batch_size: number of sentences in a mini-batch
        """
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # link: https://www.coursera.org/learn/nlp-sequence-models/lecture/ftkzt/recurrent-neural-network-model
        # Note: gamma_gate = sigmoid([a,x] * W + b)
        # 'a' of shape[batch_size, hidden_size]
        # 'x' of shape[batch_size, input_size]
        # [a,x] has shape[batch_size, hidden_size + input_size]
        self.weight_update = tf.Variable(tf.truncated_normal(shape=[hidden_size + input_size, hidden_size], seed=1))
        self.weight_forget = tf.Variable(tf.truncated_normal(shape=[hidden_size + input_size, hidden_size], seed=2))
        self.weight_candidate = tf.Variable(tf.truncated_normal(shape=[hidden_size + input_size, hidden_size], seed=3))
        self.weight_output = tf.Variable(tf.truncated_normal(shape=[hidden_size + input_size, hidden_size], seed=4))

        self.bias_update = tf.Variable(tf.truncated_normal(shape=[batch_size, hidden_size], seed=1))
        self.bias_forget = tf.Variable(tf.truncated_normal(shape=[batch_size, hidden_size], seed=2))
        # self.bias_forget = tf.Variable(tf.ones(shape=[batch_size, hidden_size]))
        self.bias_candidate = tf.Variable(tf.truncated_normal(shape=[batch_size, hidden_size], seed=3))
        self.bias_output = tf.Variable(tf.truncated_normal(shape=[batch_size, hidden_size], seed=4))


    def run_step(self, x, c, a):
        """
        Run step t
        :param x: input at time step t (in this case, batch of word2vec)
        :param c: cell state from previous step, aka c<t-1>
        :param a: hidden state from previous step, aka a<t-1>
        :return: tuple of tensors (new_cell_state, new_hidden_state)
        """
        concat_matrix = tf.concat(
            [a, x], axis=1,
            #name='concatenate'
        )
        # Note: shape[0] of matrix will be (hidden_size + input_dimension)
        # --> tf.concat([a, x], axis=1) and not else

        # new cell state candidate
        candidate = tf.tanh(
            tf.matmul(concat_matrix, self.weight_candidate) + self.bias_candidate,
            #name='create_candidate'
        )

        # forget gate
        gamma_f = tf.sigmoid(
            tf.matmul(concat_matrix, self.weight_forget) + self.bias_forget,
            #name='forget_gate'
        )

        # update gate
        gamma_u = tf.sigmoid(
            tf.matmul(concat_matrix, self.weight_update) + self.bias_update,
            #name='update_gate'
        )

        # output gate
        gamma_o = tf.sigmoid(
            tf.matmul(concat_matrix, self.weight_output) + self.bias_output,
            #name='output_gate'
        )

        # compute cell state at step t (this step)
        c_new = tf.add(
            x=tf.multiply(gamma_u, candidate), y=tf.multiply(gamma_f, c),
            #name='c_t'
        )

        # compute hidden state at step t
        a_new = tf.multiply(
            x=gamma_o, y=tf.tanh(c_new),
            #name='a_t'
        )

        return c_new, a_new


class EncoderBasic:
    """
    Uni-direction Encoder with a single LSTM cell
    """
    def __init__(self, lstm_cell):
        self.lstm_cell = lstm_cell


    def encode(self, batch_of_sentences):
        """
        Encode the sentence to vector represent
        :param batch_of_sentences: batch of source sentences (in this case, each sentence is a list of word indices)
        :return: tuple of (cell_state, hidden_states)
            - hidden_states: list of hidden_state at each time step
        """
        cell_state = tf.zeros([self.lstm_cell.batch_size, self.lstm_cell.hidden_size])
        hidden_state = tf.zeros([self.lstm_cell.batch_size, self.lstm_cell.hidden_size])
        sentence_length = tf.shape(batch_of_sentences)[1]  # length of a sentence
        hidden_states = tf.TensorArray(tf.float32, size=sentence_length, dynamic_size=True, clear_after_read=False,
                                       infer_shape=False)
        def cond(i, *_):
            return tf.less(i, sentence_length)
        def body(i, c, hid_states):
            x = batch_of_sentences[:, i]
            x = tf.map_fn(lambda e: tf.one_hot(e, len(vocab_src), dtype=tf.float32), x, dtype=tf.float32)
            if i == 0:
                c, new_hs = self.lstm_cell.run_step(x, c, hidden_state)
            else:
                c, new_hs = self.lstm_cell.run_step(x, c, hid_states.read(i - 1))
            hid_states = hid_states.write(i, new_hs)
            return i+1, c, hid_states
        _, _, hidden_states = tf.while_loop(cond, body, [0, cell_state, hidden_states])

        return hidden_states.stack()


class DecoderBasic:
    def __init__(self, lstm_cell, tgt_vocab_size):
        self.lstm_cell = lstm_cell
        self.weight_score = tf.Variable(tf.truncated_normal([lstm_cell.hidden_size, tgt_vocab_size]))
        self.bias_score = tf.Variable(tf.zeros([lstm_cell.batch_size, tgt_vocab_size]))


    def decode(self, labels, hidden_state):
        """
        Decode the represent vector from encoder (in this case, it's hidden_states[-1]).
        Inspired by Vanilla seq2seq model at link: https://guillaumegenthial.github.io/sequence-to-sequence.html
        :param labels: batch of label sentences
        whose shape is [batch_size, sentence_length] and end with <eos>
        :param hidden_state: last hidden state produced by encoder
        :return: tuple of (cell_state, logits)
        """
        cell_state = tf.zeros([self.lstm_cell.batch_size, self.lstm_cell.hidden_size])
        sentence_length = tf.shape(labels)[1]
        hidden_states = tf.TensorArray(tf.float32, size=sentence_length, dynamic_size=True, clear_after_read=False,
                                       infer_shape=False)
        logits = tf.TensorArray(tf.float32, size=sentence_length, dynamic_size=True, clear_after_read=False,
                                infer_shape=False)  # model's predictions
        i = 0
        # feed <sos> to generate first word
        cell_state, hidden_state = self.lstm_cell.run_step(
            [tf.one_hot(DataHelper.sos_id, len(vocab_tgt), dtype=tf.float32)] * self.lstm_cell.batch_size,
            cell_state, hidden_state)
        hidden_states = hidden_states.write(i, hidden_state)
        score_vector = tf.add(
            tf.matmul(hidden_state, self.weight_score), self.bias_score
        )
        logits = logits.write(i, score_vector)
        def cond(i, *_):
            return tf.less(i, sentence_length - 1)  # shift left by 1
        def body(i, c, predicts, hid_states):
            y = labels[:, i]
            y = tf.map_fn(lambda e: tf.one_hot(e, len(vocab_tgt), dtype=tf.float32), y, dtype=tf.float32)
            c, new_hs = self.lstm_cell.run_step(y, c, hid_states.read(i-1))
            hid_states = hid_states.write(i, new_hs)
            score = tf.add(
                tf.matmul(new_hs, self.weight_score), self.bias_score
            )
            predicts = predicts.write(i, score)
            return i+1, c, predicts, hid_states

        _, _, logits, hidden_states = tf.while_loop(cond, body, [i, cell_state, logits, hidden_states])

        return logits.stack(), hidden_states.stack()


def create_dataset(sentences_as_ids):
    def generator():
        for sentence in sentences_as_ids:
            yield sentence
    dataset = tf.data.Dataset.from_generator(generator, output_types=tf.int64)
    return dataset


# dataset
src_set = create_dataset(samples)
tgt_set = create_dataset(labels)
train_set = tf.data.Dataset.zip((src_set, tgt_set))
train_set = train_set.shuffle(10000)
train_set = train_set.apply(tf.contrib.data.padded_batch_and_drop_remainder(64, ([None], [None])))
it = train_set.make_initializable_iterator()
batch_x, batch_y = it.get_next()
# graph
encoder = EncoderBasic(LSTMcell(hidden_size, len(vocab_src), batch_size))
decoder = DecoderBasic(LSTMcell(hidden_size, len(vocab_tgt), batch_size), len(vocab_tgt))
encode_hidden_states = encoder.encode(batch_x)
logits = decoder.decode(batch_y, encode_hidden_states[-1])
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_y, logits=logits))
params = tf.trainable_variables()
gradients = tf.gradients(loss, params)  # derivation of loss by params
max_gradient_norm = 1.25
clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
global_step = 0
optimizer = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
