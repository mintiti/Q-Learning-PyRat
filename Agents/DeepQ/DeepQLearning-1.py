import tensorflow as tf
import PyratEnv
import random as rd
import numpy as np
from collections import deque
from tqdm import tqdm
import os.path

# CONTANTS
BUFFER_SIZE = 3000
NUMBER_EPISODES = 1000


class DeepQLearning:
    """
    First DeepQLearning try
    Only implements the target network & simple experience replay
    """

    def __init__(self, env, alpha=0.99, gamma=0.9, epsilon=0.99, buffer_size=BUFFER_SIZE):
        self.nb_episode = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.maze_dimension = env.maze_dimension
        self.env = env
        self.memory = deque(maxlen=buffer_size)

        # Initialize the NNs
        self.nn_prediction = None
        self._initialize_prediction_nn()
        self.nn_target = None
        self._initialize_target_nn()

    @classmethod
    def fromNN(cls,path):


    def save_NN(self,path):
        found = False
        number = 1
        save_name = f"neural_network_save_{number}"
        while not found:
            if os.path.exists("save/"):
                pass

    def _initialize_target_nn(self):
        nb_parameters_matrix = self.maze_dimension[0] * self.maze_dimension[1]
        self.nn_target = tf.keras.Sequential()
        self.nn_target.add(tf.keras.layers.Dense(128, activation='relu'))
        self.nn_target.add(tf.keras.layers.Dense(64, activation='relu'))
        self.nn_target.add(tf.keras.layers.Dense(32, activation='relu'))
        self.nn_target.add(tf.keras.layers.Dense(4))

        self.nn_target.compile(optimizer='adam', loss='mse')

    def _initialize_prediction_nn(self):
        nb_parameters_matrix = self.maze_dimension[0] * self.maze_dimension[1]
        self.nn_prediction = tf.keras.Sequential()
        self.nn_prediction.add(tf.keras.layers.Dense(128, activation='relu'))
        self.nn_prediction.add(tf.keras.layers.Dense(64, activation='relu'))
        self.nn_prediction.add(tf.keras.layers.Dense(32, activation='relu'))
        self.nn_prediction.add(tf.keras.layers.Dense(4))

        self.nn_prediction.compile(optimizer='adam', loss='mse')

    def update_target_nn(self):
        self.nn_target.set_weights(self.nn_prediction.get_weights())

    def train_prediction(self, inputs, targets):
        self.nn_prediction.fit(x=inputs, y=targets, batch_size=32)

    def act(self, state):
        if rd.uniform(0, 1) <= self.epsilon:
            return self.env.action_space.sample()[0]
        act_values = self.nn_prediction.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.appendleft((state, action, reward, next_state, done))

    # TODO
    def replay(self, batch_size):
        minibatch = np.array(rd.sample(self.memory, batch_size))
        for state, action, r, next_state, d in minibatch:
            target = r
            if not d:
                target = r + self.gamma * np.amax(self.nn_target.predict(next_state))
            target_f = self.nn_prediction.predict(state)
            target_f[0][action] = target
            self.nn_prediction.fit(state, target_f, batch_size=1, epochs=1)
        if self.epsilon > 0.05:
            self.epsilon *= 0.99


def flatten_state(non_flattened_state):
    r = []
    for i in non_flattened_state[0].flatten():
        r.append(i)
    r.append(non_flattened_state[1])
    for i in non_flattened_state[2]:
        r.append(i)
    for i in non_flattened_state[3]:
        r.append(i)
    for i in non_flattened_state[4]:
        r.append(i)
    r = np.array(r)
    return np.reshape(r, (1, len(r)))


def _non_flattened_state_from_obs(obs, player):
    r = [obs['pieces_of_cheese'], obs['turn']]
    player1_score, player2_score = obs['player_scores']
    player1_location = obs['player1_location']
    player2_location = obs['player2_location']
    if player == 1:
        r.append((player1_score, player2_score))
        r.append(player1_location)
        r.append(player2_location)

    elif player == 2:
        r.append((player2_score, player1_score))
        r.append(player2_location)
        r.append(player1_location)
    return r


def flattened_state_from_obs(obs, player):
    non_flattened_r = _non_flattened_state_from_obs(obs, player)
    return flatten_state(non_flattened_r)


def test_agents(agent1, agent2):
    pass


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import time

    maze_file = "..\\..\\maze_files\\maze.p"
    dirname = os.path.dirname(os.path.abspath(__file__))
    maze_file = os.path.join(dirname, maze_file)

    # Initialize environment
    if os.path.exists(maze_file):
        env = PyratEnv.PyratEnv.fromPickle(maze_file)
    else:
        env = PyratEnv.PyratEnv()
        env.save_pickle(maze_file)

    # Initialize players
    player1 = DeepQLearning(env)
    player2 = DeepQLearning(env)

    # Tracking variables
    player1_scores = []
    player2_scores = []

    for episode in tqdm(range(NUMBER_EPISODES)):
        env = PyratEnv.PyratEnv.fromPickle(maze_file)
        state1 = [env.cheese_matrix, 0, (env.player1_score, env.player2_score), env.player1_location,
                  env.player2_location]
        state2 = [env.cheese_matrix, 0, (env.player2_score, env.player1_score), env.player2_location,
                  env.player1_location]
        state1 = flatten_state(state1)
        state2 = flatten_state(state2)

        done = False
        turn = 0
        while not done:
            # Decide actions
            decision1 = player1.act(state1)
            decision2 = player2.act(state2)
            # print (decision1, decision2)

            # Get the next state and rewards
            obs, reward, done, _ = env.step((decision1, decision2))
            next_state1 = flattened_state_from_obs(obs, 1)
            next_state2 = flattened_state_from_obs(obs, 2)

            # remember the transition
            player1.remember(state1, decision1, reward, next_state1, done)
            player2.remember(state2, decision2, - reward, next_state2, done)

            # make the next state the new state for the next turn
            state1 = next_state1
            state2 = next_state2
            # print(f'Done turn {turn} of episode {episode}')

            turn += 1

            if done:
                score1 = env.player1_score
                score2 = env.player2_score
                player1_scores.append(score1)
                player2_scores.append(score2)
                print(f"""Done episode {episode} / {NUMBER_EPISODES}
    Player 1 score : {score1}
    Player 2 score : {score2}
    Epsilon value : {player1.epsilon}
                """)
                break
        player1.replay(32)
        player2.replay(32)

        if episode % 5 == 0:
            # hardcopy prediction to target
            player1.update_target_nn()
            player2.update_target_nn()

    episodes = [i for i in range(len(player1_scores))]
    plt.plot(episodes, player1_scores, 'r')  # plotting t, a separately
    plt.plot(episodes, player2_scores, 'b')  # plotting t, b separately
    plt.show()
