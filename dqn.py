from collections import deque, namedtuple
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution3D, MaxPooling3D
from keras.layers import Dropout, Flatten, Merge
import numpy as np
import theano
from theano import tensor as t
import tables


def conv_net(n_kind, n_in, n_out, n_hidden):
    branches = [Sequential() for _ in range(n_kind)]

    def double_conv(model, n_hidden):
        model.add(Convolution3D(n_hidden, 3, 3, 3, border_mode='valid',
                                input_shape=(n_in)))
        model.add(Activation('relu'))
        # model.add(Convolution3D(n_hidden, 3, 3, 3))
        # model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(Dropout(0.25))

    def dense(model, n_hidden):
        model.add(Dense(n_hidden))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

    for model in branches:
        double_conv(model=model, n_hidden=n_hidden[0])
        double_conv(model=model, n_hidden=n_hidden[1])
        model.add(Flatten())
        dense(model=model, n_hidden=n_hidden[2])

    merged = Merge(branches, mode='concat')
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(n_out))
    return final_model


def loss(y_true, y_pred):
    """y_true[0] is action. y_true_[1] is Q-value"""
    a = y_true[:, 0]
    q_t = y_true[:, 1]
    q_y = y_pred[t.arange(a.shape[0]), t.cast(a, 'int32')]
    return -t.mean(t.sqr(q_t - q_y))


def model(n_kind, n_in, n_out, n_hidden):
    model = conv_net(n_kind, n_in, n_out, n_hidden)
    model.compile(optimizer='adam', loss=loss)
    return model


"""state, action, next_state, reward"""
Experiment = namedtuple('Experiment', ['s', 'a', 'n', 'r'])


class replay_memory(deque):
    def __init__(self, n_store, memory="replay.mm"):
        self.n_store = n_store
        self.memory = open(memory, 'a')

    def append(self, s, a, n, r):
        super().append(Experiment(s, a, n, r))
        if len(self) > self.n_store:
            pop = self.popleft()
            self.memory.write(pop)

    def actions(self):
        return np.array([e.a for e in self])

    def rewards(self):
        return np.array([e.r for e in self])


def epsilon_greedy(epsilon, qs):
    rand = np.random.random()
    l = qs.shape[1]
    if epsilon > rand:
        return np.random.choice(np.arange(l), size=(qs.shape[0],))
    else:
        return np.argmax(qs, axis=1)


if __name__ == '__main__':
    print(epsilon_greedy(1, np.zeros((10, 29))))
    a = np.array([[i / j for i in range(1, 29)] for j in range(1, 10)]).astype(
        np.float32)
    print(a)
    print('')
    c = t.fmatrix('c')
    b = t.argmax(c, axis=1)
    f = theano.function([c], b)
    print(f(a))
    a = model(3, (4, 16, 16, 16), 10, (25, 25, 100))
    spam = np.zeros((1, 4, 16, 16, 16))
    print(epsilon_greedy(1, a.predict([spam, spam, spam])))
    print(epsilon_greedy(0, a.predict([spam, spam, spam])))
