from __future__ import unicode_literals, print_function, division
import random

import glob
import string
import unicodedata
from io import open

import torch
import torch.nn as nn
from torch.autograd import Variable

import time
import math

import matplotlib.pyplot as plt

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker


def findFiles(path): return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# Build the category_lines dictionary, a list of names per language
category_lines_training = {}
category_lines_values = {}
all_categories = []

for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    idx = (int)(len(lines) * .7)
    category_lines_training[category] = lines[:idx]
    category_lines_values[category] = lines[idx:]

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')


class RNN_hidden_character_category_1_given(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_hidden_character_category_1_given, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, category):
        return Variable(torch.zeros(1, self.hidden_size))

class RNN_hidden_character_2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_hidden_character_2, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, category):
        return torch.cat((category, Variable(torch.zeros((1, self.hidden_size - n_categories)))), 1)




class RNN_hidden_category_3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_hidden_category_3, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, category):
        return Variable(torch.zeros(1, self.hidden_size))





class RNN_hidden_4(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_hidden_4, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        hidden = self.i2h(hidden)
        output = self.i2o(hidden)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, category):
        return torch.cat((category, Variable(torch.zeros((1, self.hidden_size - n_categories)))), 1)



#TODO::FM:: MIght need to remove
# def categoryFromOutput(output):
#     top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
#     category_i = top_i[0][0]
#     return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines_training[category])
    return category, line


# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = Variable(categoryTensor(category))
    input_line_tensor = Variable(inputTensor(line))
    target_line_tensor = Variable(targetTensor(line))
    return category_tensor, input_line_tensor, target_line_tensor


def mapLineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


criterion = nn.NLLLoss()
learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(category_tensor, input_line_tensor, target_line_tensor, model):
    model.zero_grad()
    loss, output = getLoss(input_line_tensor, category_tensor, target_line_tensor, model)
    loss.backward()

    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0] / input_line_tensor.size()[0]


# #TODO::FM::
# def validationExample(category, line):
#     category_tensor = Variable(categoryTensor(category))
#     line_tensor = Variable(mapLineToTensor(line))
#     target_line_tensor = Variable(targetTensor(line))
#     return category_tensor, line_tensor, target_line_tensor


def evaluate(model):
    count = 0
    loss = 0
    for category in all_categories:
        for line in category_lines_values[category]:
            category_tensor = Variable(categoryTensor(category))
            line_tensor = Variable(mapLineToTensor(line))
            target_line_tensor = Variable(targetTensor(line))
            l, output = getLoss(line_tensor, category_tensor, target_line_tensor, model)
            loss += l
            count += 1

    avg_loss = loss.data / count
    return avg_loss


def getLoss(input_line_tensor, category_tensor, target_line_tensor, model):
    hidden = model.initHidden(category_tensor)
    loss = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = model(category_tensor, input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])
    return loss, output


rnn1 = RNN_hidden_character_category_1_given(n_letters, 128, n_letters)
rnn2 = RNN_hidden_character_2(n_letters, 128, n_letters)
rnn3 = RNN_hidden_category_3(n_letters, 128, n_letters)
rnn4 = RNN_hidden_4(n_letters, 128, n_letters)

n_iters = 100000
print_every = 5000
plot_every = 500

start = time.time()
model_types = [rnn1, rnn2, rnn3, rnn4]
model_labels = ['(i) ucc', '(ii) u(ch)', '(iii)u(ca)', '(iv)u']
losses = []
# model_trainingLosses = []  TODO

for i in range(len(model_types)):

    # Keep track of losses for plotting
    current_loss = 0
    # train_losses = []   TODO
    value_loss = []
    total_loss = 0
    model_label = model_labels[i]
    model_type = model_types[i]
    print('RNN TYPE--------------> %s <---------------\n' % model_label)
    for iter in range(1, n_iters + 1):
        category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, input_line_tensor, target_line_tensor, model_type)
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            value_loss.append(evaluate(model_type))
            # train_losses.append(total_loss / plot_every)    TODO
            total_loss = 0

    losses.append(value_loss)
    # model_trainingLosses.append(train_losses)   TODO

plt.figure()
for i in range(len(model_types)):
    print('\nFor %s: Loss: %.4f' % (model_labels[i], losses[i][-1]))
    plt.plot(losses[i], label=model_labels[i])

plt.xlabel("Number of iterations (thousand(s))")
plt.ylabel("test(validation) negative log likelihood")

plt.legend()
plt.title('a4 q2 -Validation_Loss')
plt.savefig('a4-2b-validation.png')

#TODO::FM::Might need to remove this
# plt.figure()
# for i in range(len(model_types)):
#     plt.plot(model_trainingLosses[i], label=model_labels[i])
#
# plt.legend()
# plt.title('a4 q2b Train_loss')
# plt.savefig('a4-2b-train.png')
