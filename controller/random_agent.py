import numpy as np

class RandomAgent():

    def __init__(self, nx, ny, nd, num_actions, game):

        self.num_actions = num_actions
        self.game = game

        self.Q = np.zeros([nx, ny, nd, num_actions])

        self.gamma = 0.8
        self.alpha = 0.1
        self.n_episodes = 500
        self.max_iter = 50000
        self.epsilon = 0.9

    def learn(self):

        for e in range(self.n_episodes):

            self.game.reset()
            state = self.game.get_state()

            count = 0
            while count < self.max_iter:

                action = self.select_action(state)

                next_state, reward, done = self.game.step(action)
                if done:
                    break

                self.updateQ(state, action, reward, next_state)

                state = next_state
                count += 1

            self.epsilon = max(self.epsilon - 1 / (self.n_episodes - 1.), 0)

    def updateQ(self, state, action, reward, next_state):
        current_q = self.Q[state][action]
        next_q = reward + self.gamma * max(self.Q[next_state])
        self.Q[state][action] = current_q + self.alpha * (next_q - current_q) 


    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.Q[state][:])