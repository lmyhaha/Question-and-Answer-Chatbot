#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import re
import time

# from warnings import simplefilter
# simplefilter(action='ignore', category=FutureWarning)

#config = tf.ConfigProto()  
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#config.gpu_options.allow_growth = True

# Importing the dataset
lines = open('dataset/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('dataset/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# Creating a dictiorany that maps each line and its id

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
# id2line dic {Lxxx:话} 每一行说了什么
# print(id2line)
# exit()

# Creating a list of all the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))
# conversations_ids list [[Lxx,Lxx],[]] 每段对话由哪几行组成 89 59 56
# ma = 0
# for i in conversations_ids:
#     if len(i) > ma and len(i) < 59:
#         ma = len(i)
# print(ma)
# exit()

# Getting separately the questions and the answers
questions = []
answers = []
for conversations_id in conversations_ids:
    for i in range(len(conversations_id) - 1):
        question = []
        for j in range(i+1):
            question.append(id2line[conversations_id[j]])
        questions.append(question)
        answers.append(id2line[conversations_id[i + 1]])
# questions list ['xx','xx'] 问句
# answers list ['xx','xx'] 答句
# print(questions[0:10])
# print(answers[0:10])
# exit()

# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text


# Cleaning the questions
clean_questions = []
for question in questions:
    clean_question = []
    for se in question:
        clean_question.append(clean_text(se))
    clean_questions.append(clean_question)

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
# clean_questions list ['xx','xx'] 处理干净的问句
# clean_answers list ['xx','xx'] 处理干净的答句

# Filtering out the questions and answers that are too short or too long
short_questions = []
short_answers = []
i = 0
for question in clean_questions:
    flag = 1
    for se in question:
        if len(se.split()) < 2 or len(se.split()) > 25:
            flag = 0
    if flag == 1:
        if len(question) > 15:
            cl_question = question[-15:]
            short_questions.append(cl_question)
        else:
            short_questions.append(question)
        short_answers.append(clean_answers[i])
    i += 1
clean_questions = []
clean_answers = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        clean_answers.append(answer)
        clean_questions.append(short_questions[i])
    i += 1
# clean_questions list ['xx','xx'] 长度适当的处理干净的问句
# clean_answers list ['xx','xx'] 长度适当的处理干净的答句

# Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 15 + 1):
    count = 0
    for i in enumerate(clean_questions):
        if len(i[1]) == length and count < 1200:
            sorted_clean_questions.append(clean_questions[i[0]])
            sorted_clean_answers.append(clean_answers[i[0]])
            count += 1

clean_questions = sorted_clean_questions
clean_answers = sorted_clean_answers

# Creating a dictionary that maps each word to its number os ocurrences
word2count = {}
for question in clean_questions:
    for se in question:
        for word in se.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
# word2count dic {'单词':出现次数}

for i in word2count:
    if word2count[i] > 1000:
        print(i, word2count[i])
exit()

# Creating two dictionaries that map the questions words and the answers word to a unique integer
threshold_questions = 15
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_questions:
        questionswords2int[word] = word_number
        word_number += 1
threshold_answers = 15
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers:
        answerswords2int[word] = word_number
        word_number += 1
# threshold_questions 15 编入问题词典出现最低次数
# threshold_answers 15 编入答案词典出现最低次数
# questionswords2int dic {'词':编码}
# answerswords2int dic {'词':编码}

# Adding the last tokens to these 2 dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

# Creating the inverse dictionary of the answerswords2int dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}
# answersints2word dic {编码:'词'}

# Adding the end of string token to the end of every answers
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Tanslating all the questions and the answers into integers and replacing
# all the words that were filtered out by out
questions_to_int = []
for question in clean_questions:
    intss = []
    for se in question:
        ints = []
        for word in se.split():
            if word not in questionswords2int:
                ints.append(questionswords2int['<OUT>'])
            else:
                ints.append(questionswords2int[word])
        intss.append(ints)
    questions_to_int.append(intss)

answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_to_int.append(ints)
# questions_to_int list [[编码, 编码],[]] 问题转编码
# answers_to_int list [[编码, 编码],[]] 答案转编码
# print(questions_to_int[0:10])
# exit()

# Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 15 + 1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])



# sorted_clean_questions list [] 将问题按问题长短排序
# sorted_clean_answers list [] 将答案按问题长短排序
# print(sorted_clean_questions[:10])
# print(sorted_clean_answers[:10])
# exit()

######## PART 2 Building the SEQ2SEQ Model ######

# Creating placeholders for the inputs and the targets

def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, 15, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, lr, keep_prob


# Preprocessing the target

def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


# Creating the encoder RNN layer
def my_encoder(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                       cell_bw=encoder_cell,
                                                       sequence_length=sequence_length,
                                                       inputs=rnn_inputs,
                                                       dtype=tf.float32)
    return encoder_state[0][2][1]
# ValueError: Variable bidirectional_rnn/fw/multi_rnn_cell/cell_0/basic_lstm_cell/weights

def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    outputs = list()
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    with tf.variable_scope('RNN'):
        for timestep in range(rnn_inputs.shape[1]):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            # 这里的state保存了每一层 LSTM 的状态
            _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                               cell_bw=encoder_cell,
                                                               sequence_length=sequence_length,
                                                               inputs=rnn_inputs[:, timestep],
                                                               dtype=tf.float32)
            outputs.append(encoder_state[0][2][1])
    outputs = tf.stack(outputs, 1)
    # print(outputs)
    # print(outputs.shape)
    # print(sequence_length)
    question_length = tf.placeholder_with_default(15, None, name='question_length')

    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                       cell_bw=encoder_cell,
                                                       sequence_length=question_length,
                                                       inputs=outputs,
                                                       dtype=tf.float32)
    return encoder_state

    rnn_sec_inputs = []
    for i in range(rnn_inputs.shape[1]):
        rnn_fir_inputs = rnn_inputs[:, i]
        rnn_sec_inputs.append(my_encoder(rnn_fir_inputs, rnn_size, num_layers, keep_prob, sequence_length))
        # lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)  # 基础LSTM单元中的神经元数量
        # lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)  # dropout网络中每个单元在每次有数据流入时以一定的概率(keep prob)正常工作，否则输出0值
        # encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)  # 将多个BasicLSTMCell单元汇总为一个
        #
        # print(i)
        # _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
        #                                                    cell_bw=encoder_cell,
        #                                                    sequence_length=sequence_length,
        #                                                    inputs=rnn_fir_inputs,
        #                                                    dtype=tf.float32)
        # rnn_sec_inputs.append(encoder_state[0][2][1])
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                       cell_bw=encoder_cell,
                                                       sequence_length=sequence_length,
                                                       inputs=rnn_sec_inputs,
                                                       dtype=tf.float32)
    return encoder_state


# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input,
                        sequence_length, decoding_scope, output_function,
                        keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
        attention_states,
        attention_option='bahdanau',
        num_units=decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name='attn_dec_train')
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
        decoder_cell,
        training_decoder_function,
        decoder_embedded_input,
        sequence_length,
        scope=decoding_scope)
    decoder_output_drop = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_drop)


# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embedded_matrix,
                    sos_id, eos_id, maximum_length, num_words,
                    decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
        attention_states,
        attention_option='bahdanau',
        num_units=decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embedded_matrix, sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name='attn_dec_inf')
    test_prediction, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
        decoder_cell,
        test_decoder_function,
        scope=decoding_scope)
    return test_prediction


# Creating the decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embedded_matrix, encoder_state, num_words,
                sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope=decoding_scope,
                                                                      weights_initializer=weights,
                                                                      biases_initializer=biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embedded_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
        return training_predictions, test_predictions


# Building the Seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words,
                  encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer=tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn_layer(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embedded_matrix = tf.Variable(tf.random_uniform([(questions_num_words + 1), decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embedded_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, decoder_embedded_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


######### Part 3 Training the Seq2seq Model ######

# Setting the hyperparameters
epochs = 100
batch_size = 32
rnn_size = 512
num_layers = 3
encoding_embedding_size = 1024
decoding_embedding_size = 1024
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining a Session
tf.reset_default_graph()
#session = tf.InteractiveSession(config = config)
session = tf.InteractiveSession()
# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name='sequence_length')

# Setting the shape of the inputs tensor
input_shape = tf.shape(inputs)

# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)
print(test_predictions)
print('----------------------------------')
# Setting up the Loss Error, the optimizer and Gradient clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in
                         gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)


# Padding the sequence with the <PAD> token
def apply_padding_questions(batch_of_sequences, word2int):
    max_question_length = 15  # max([len(question) for question in batch_of_sequences])
    max_sequence_length = max([max([len(sequence) for sequence in question]) for question in batch_of_sequences])
    # print(max_question_length, max_sequence_length)
    return [[sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in question] + [[word2int['<PAD>']] * max_sequence_length] * (max_question_length - len(question)) for question in batch_of_sequences]


def apply_padding_answers(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index: start_index + batch_size]
        answers_in_batch = answers[start_index: start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding_questions(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding_answers(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch


# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(
            split_into_batches(training_questions,
                               training_answers,
                               batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping,
                                                    loss_error], {inputs: padded_questions_in_batch,
                                                                  targets: padded_answers_in_batch,
                                                                  lr: learning_rate,
                                                                  sequence_length: padded_answers_in_batch.shape[1],
                                                                  keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print(
                'Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 batches: {:d} seconds'.format(
                    epoch, epochs,
                    batch_index,
                    len(training_questions) // batch_size,
                    total_training_loss_error / batch_index_check_training_loss,
                    int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(
                    split_into_batches(validation_questions,
                                       validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[
                                                                           1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(
                average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better Now!!!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print('Sorry I do not speak better, I need to practice more')
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print('My apologies, I cannot speak better anymore, this is the best I can do')
        break

print('Over')
