from PyratEnv import PyratEnv
import pickle
import random
import numpy as np

# CONSTANTS
EPS_DECAY = 0.99
EPISODES = 100


def flatten_state(state):
    returnL = []
    for i in state[0]:
        returnL.append(i)
    for i in state[1]:
        returnL.append(i)
    for i in state[2].flatten():
        returnL.append(i)

    return tuple(returnL)


class QLearningAgent:
    """
    Naive Q-Learning agent
    Inputs :
        unnamed :
            - env : the environment you wanna play the agent on
        named :
            - player : player 1 or player 2 in the game
                    player == 1 : rat
                    player == 2 = python
            - alpha : the starting learning rate
            - gamma : the starting discount factor
            - epsilon : the starting explore rate (for the epsilon-greedy policy)
            - epsilon_min : the minimum epsilon you want to reach

    Interface methods :
        - act : acts on a given observation on the epsilon-greedy policy
        - update_q_value : updates the q value of the q table
    """

    def __init__(self, env, player=1, alpha=0.8, gamma=0.8, epsilon=0.99, epsilon_min=0.5):
        self.q_table = dict()
        self.nb_episode = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.epsilon_min = epsilon_min
        self.player = player

    def act(self, obs):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()[0]
        else:
            state_dict_key = self._key_from_obs(obs)
            action, max_value = self._get_max_action(state_dict_key)
            return action

    def update_q_value(self, old_obs, obs, decision, reward):
        new_state_dict_key = self._key_from_obs(obs)
        old_state_dict_key = self._key_from_obs(old_obs)
        _, next_max = self._get_max_action(new_state_dict_key)
        if self.player == 2:
            reward = -reward
        self.q_table[old_state_dict_key][decision] = (1 - self.alpha) * self.q_table[old_state_dict_key][decision] + self.alpha * \
                                                     (reward + naive_QLearn1.gamma * next_max)

    def _get_max_action(self, state_dict_key):
        if state_dict_key not in self.q_table:
            self.q_table[state_dict_key] = [0, 0, 0, 0]
        action = None
        q_list = self.q_table[state_dict_key]
        m = max(q_list)
        actions = []
        for i in range(len(q_list)):
            if q_list[i] == m:
                actions.append(i)

        return random.choice(actions), m

    def parameter_tune(self):
        if self.epsilon_min < self.epsilon:
            self.epsilon *= EPS_DECAY

    def _key_from_obs(self, obs):
        r = []
        player1_location = obs['player1_location']
        player2_location = obs['player2_location']
        pieces_of_cheese = obs['pieces_of_cheese']
        if self.player == 1:
            for i in player1_location:
                r.append(i)
            for i in player2_location:
                r.append(i)
        else:
            for i in player2_location:
                r.append(i)
            for i in player1_location:
                r.append(i)
        for i in pieces_of_cheese.flatten():
            r.append(i)
        return tuple(r)


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # import numpy as np
    # Initialize the env and the agent
    maze_file = "D:\\Users\\Minh Tri Truong\\Documents\\IMT\\Projets ML\\Q-Learning-PyRat\\maze_files\\maze.p"
    env = PyratEnv.fromPickle(maze_file)
    naive_QLearn1 = QLearningAgent(env)
    naive_QLearn2 = QLearningAgent(env)

    # Iterate over nb of episodes
    for episode in range(EPISODES):
        # see the scores of both players
        player1_scores = [0]
        player2_scores = [0]

        # Reset the environment
        env = PyratEnv.fromPickle(maze_file)
        done = False

        # Generate the states for the 2 players
        # The locations are just reversed
        # The reward received is for player 1, player 2 gets - reward
        state1 = (env.player1_location, env.player2_location, env.cheese_matrix)
        state2 = (env.player2_location, env.player1_location, env.cheese_matrix)

        i = 0
        while not done:
            decision1 = naive_QLearn1._get_action(state1, env)
            decision2 = naive_QLearn2._get_action(state2, env)
            action = (decision1, decision2)
            # remember the old Q values

            obs, reward, done, _ = env.step(action)

            # make the next states
            next_player1_location = obs['player1_location']
            next_player2_location = obs['player2_location']
            next_cheese_matrix = obs['pieces_of_cheese']
            new_state1 = (next_player1_location, next_player2_location, next_cheese_matrix)
            new_state2 = (next_player2_location, next_player1_location, next_cheese_matrix)
            ## agent.update_q_value(obs, decision1)
            # Update the q value for player1
            old_value1 = naive_QLearn1.get_q_value(state1, decision1)
            _, next_max = naive_QLearn1._get_max_action(flatten_state(new_state1))
            new_value1 = (1 - naive_QLearn1.alpha) * old_value1 + naive_QLearn1.alpha * (
                    reward + naive_QLearn1.gamma * next_max)
            naive_QLearn1.q_table[flatten_state(state1)][decision1] = new_value1

            # Update the q value for player2
            old_value2 = naive_QLearn2.get_q_value(state2, decision2)
            _, next_max = naive_QLearn2._get_max_action(flatten_state(new_state2))
            new_value2 = (1 - naive_QLearn2.alpha) * old_value2 + naive_QLearn2.alpha * (
                    (- reward) + naive_QLearn2.gamma * next_max)
            naive_QLearn2.q_table[flatten_state(state2)][decision2] = new_value2

            # add scores to the scores list
            score1, score2 = obs['player_scores']
            player1_scores.append(score1)
            player2_scores.append(score2)

            state1 = new_state1
            state2 = new_state2

            # print(f'turn {i}/2000')
            i += 1

            if done:
                # steps = [i for i in range(len(player2_scores))]
                # plt.plot(steps,player1_scores, 'r')
                # plt.plot(steps,player2_scores, 'b')
                # plt.show()
                print(f"""Done episode {episode} / {EPISODES}
    Player 1 score : {score1}
    Player 2 score : {score2}
    Epsilon value : {naive_QLearn1.epsilon}
""")
                naive_QLearn2.parameter_tune()
                naive_QLearn1.parameter_tune()
    print(f"""player 1 q_table : {naive_QLearn1.q_table}
player2 q_table : {naive_QLearn2.q_table}""")
