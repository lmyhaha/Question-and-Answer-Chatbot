import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import re
import time


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
        if len(i[1]) == length and count < 600:
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

epochs = 100
batch_size = 1
rnn_size = 512
num_layers = 3
encoding_embedding_size = 1024
decoding_embedding_size = 1024
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

print('loading......')
# Loading the wights and running the session
# checkpoint = './chatbot_weights.ckpt'
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.restore(session, checkpoint)

saver = tf.train.import_meta_graph('./chatbot_weights.ckpt.meta')
saver.restore(session, './chatbot_weights.ckpt')

# Converting the question from strings to lists of encoding integers
def convert_string2int(questions, word2int):
    question = questions[-1]
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]

count = 0
his = []
# Setting up the chatbot
while (True):
    question = input("You: ")
    his.append(question)
    if question == 'Goodbye' or len(his)>15:
        break
    question0 = [questionswords2int['<PAD>']] * (25)
    question = []
    for ques in his:
        ques = convert_string2int(ques, questionswords2int)
        ques = ques + [questionswords2int['<PAD>']] * (25 - len(ques))
        question.append(ques)
    question = question + [question0] * (15-len(his))
    fake_batch = np.zeros((batch_size, 15, 25))
    fake_batch[0] = question
    predicted_answer = session.run(tf.get_default_graph().get_tensor_by_name("decoding/dynamic_rnn_decoder_1/transpose:0"), feed_dict={tf.get_default_graph().get_tensor_by_name("input:0"): fake_batch, tf.get_default_graph().get_tensor_by_name("keep_prob:0"): 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersints2word[i] == 'i':
            token = 'I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)
    his.append(answer)