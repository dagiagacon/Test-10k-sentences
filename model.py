import tensorflow as tf
import data as DataHelper

samples, labels = DataHelper.get_train_set()
test_samples, test_labels = DataHelper.get_test_set()
vocab_src, vocab_tgt = DataHelper.get_vocabs()


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
        self.weight_update = tf.truncated_normal(shape=[hidden_size + input_size, hidden_size], seed=1)
        self.weight_forget = tf.truncated_normal(shape=[hidden_size + input_size, hidden_size], seed=2)
        self.weight_candidate = tf.truncated_normal(shape=[hidden_size + input_size, hidden_size], seed=3)
        self.weight_output = tf.truncated_normal(shape=[hidden_size + input_size, hidden_size], seed=4)

        self.bias_update = tf.truncated_normal(shape=[batch_size, hidden_size], seed=1)
        self.bias_forget = tf.truncated_normal(shape=[batch_size, hidden_size], seed=2)
        # self.bias_forget = tf.ones(shape=[batch_size, hidden_size])
        self.bias_candidate = tf.truncated_normal(shape=[batch_size, hidden_size], seed=3)
        self.bias_output = tf.truncated_normal(shape=[batch_size, hidden_size], seed=4)


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
        hidden_states = []
        sentence_length = len(batch_of_sentences[0])  # length of a sentence
        for i in range(sentence_length):
            x = batch_of_sentences[:, i]
            cell_state, hidden_state = self.lstm_cell.run_step(x, cell_state, hidden_state)
            hidden_states.append(hidden_state)
        return hidden_states


class DecoderBasic:
    def __init__(self, lstm_cell, tgt_vocab_size):
        self.lstm_cell = lstm_cell
        self.weight_score = tf.truncated_normal([lstm_cell.hidden_size, tgt_vocab_size])
        self.bias_score = tf.zeros([lstm_cell.batch_size, tgt_vocab_size])


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
        sentence_length = len(labels[0])
        logits = []  # model's predictions
        # feed <sos> to generate first word
        cell_state, hidden_state = self.lstm_cell.run_step([vocab_tgt[DataHelper.sos_id]] * self.lstm_cell.batch_size,
                                                           cell_state, hidden_state)
        score_vector = tf.add(
            tf.matmul(hidden_state, self.weight_score), self.bias_score
        )
        logits.append(score_vector)

        for i in range(sentence_length - 1): # shift by 1
            x = labels[:, i]
            x = [self.embeddings[ids] for ids in x]  # transform to batch of vector
            cell_state, hidden_state = self.lstm_cell.run_step(x, cell_state, hidden_state)
            score_vector = tf.add(
                tf.matmul(hidden_state, self.weight_score), self.bias_score
            )
            logits.append(score_vector)

        return logits
