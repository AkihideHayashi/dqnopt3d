import dqn
import mol
import numpy as np
import copy
import ase
from ase.io import xyz


class Main(object):
    def __init__(self, atoms, alpha, gamma, save_param='params.hdf5'):
        n_eyes = 3
        self.epsilon = 0.3
        self.field = mol.Field(atoms)
        self.model = dqn.model(n_kind=n_eyes, n_in=(5, 16, 16, 16),
                               n_out=25, n_hidden=(25, 25, 100))
        self.replay_memory = dqn.replay_memory(n_store=96)
        self.alpha = alpha
        self.gamma = gamma
        self.state = self.field.make_sights()
        self.reward = self.field.calc()
        self.atom_saver = open("atoms.xyz", 'a')

    def learn(self):
        states = self.replay_memory.states()
        next_states = self.replay_memory.next_states()
        qs = self.model.predict(states)
        next_qs = self.model.predict(next_states)
        a = self.replay_memory.actions()
        r = self.replay_memory.rewards()

        selected_qs = qs[np.arange(a.shape[0]), a]
        max_next_qs = np.argmax(next_qs, axis=1)
        new_qs = selected_qs + self.alpha * (
            r + self.gamma * max_next_qs - selected_qs)
        y = np.concatenate([a, new_qs], axis=1)
        self.model.fit(states, y, nb_epoch=10, batch_size=32)
        self.model.save_weights('params.hdf5')

    def signal(self, signal):
        if signal < 24:
            if signal % 6 == 0:
                direction = np.array([0.1, 0., 0.])
            elif signal % 6 == 1:
                direction = np.array([-0.1, 0., 0.])
            elif signal % 6 == 2:
                direction = np.array([0., 0.1, 0.])
            elif signal % 6 == 3:
                direction = np.array([0., -0.1, 0.])
            elif signal % 6 == 4:
                direction = np.array([0., 0., 0.1])
            else:
                direction = np.array([0., 0., -0.1])
            if signal // 6 == 0:
                self.field.hand.r += direction
            elif signal // 6 == 1:
                self.field.cubes[0].center += direction
            elif signal // 6 == 2:
                self.field.cubes[1].center += direction
            elif signal // 6 == 3:
                self.field.cubes[2].center += direction
            self.state = self.field.make_sights()
        else:
            self.field.pull()
            self.state = self.field.make_sights()
            self.reward = self.field.calc()

    def action(self):
        xyz.write_xyz(self.atom_saver, self.field.atoms)
        state = self.state
        s = list(map(lambda s: s[np.newaxis, :], state))
        action = dqn.epsilon_greedy(self.epsilon, self.model.predict(s))
        self.signal(action)
        next_state = self.state
        reward = self.reward
        self.replay_memory.append(state, action, next_state, reward)


def main():
    atoms = ase.Atom('HH', [[0, 0, 0], [1, 1, 1]])
    game = Main(atoms, 1, 1)
    for i in range(10000):
        game.action()
        if i % 10 == 9:
            game.learn()


if __name__ == '__main__':
    atoms = ase.Atoms('HH', [[0, 0, 0], [1, 1, 1]])
    game = Main(atoms, 1, 1)
    game.action()
    game.action()
    game.action()
    game.learn()
    print(game.field.atoms)
