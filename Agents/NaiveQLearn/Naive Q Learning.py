from PyratEnv import PyratEnv
import pickle
import random
#CONSTANTS
EPS_DECAY = 0.995
EPISODES  = 1000

def flatten_state(state):
    returnL = []
    for i in state[0]:
        returnL.append(i)
    for i in state[1]:
        returnL.append(i)
    for i in state[2].flatten():
        returnL.append(i)

    return tuple(returnL)


class QLearningAgent :


    def __init__(self, env,alpha = 0.8, gamma = 0.8, epsilon = 0.99,epsilon_min = 0.5):
        self.q_table = dict()
        self.nb_episode = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.epsilon_min = epsilon_min


    def get_action(self,state,env):
        state = flatten_state(state)
        action = None
        if random.uniform(0,1) < self.epsilon:
            action = env.action_space.sample()[0]
        else :
            action, _ = self.get_max_action(state)
        return action

    def get_max_action(self,state):
        if state not in self.q_table :
            self.q_table[state] = [0,0,0,0]
        action = None
        q_list = self.q_table[state]
        m = max(q_list)
        actions = []
        for i in range(len(q_list)):
            if q_list[i] == m :
                actions.append(i)

        return random.choice(actions), m

    def get_q_value(self,state,action):
        state = flatten_state(state)
        if state not in self.q_table :
            self.q_table[state] = [0,0,0,0]
        return self.q_table[state][action]

    def parameter_tune(self):
        if self.epsilon_min < self.epsilon :
            self.epsilon *= EPS_DECAY

if __name__ == '__main__':
    #import matplotlib.pyplot as plt
    #import numpy as np
    # Initialize the env and the agent
    maze_file = "D:\\Users\\Minh Tri Truong\\Documents\\IMT\\Projets ML\\Q-Learning-PyRat\\maze_files\\maze.p"
    env = PyratEnv.fromPickle(maze_file)
    naive_QLearn1 = QLearningAgent(env)
    naive_QLearn2 = QLearningAgent(env)

    # Iterate over nb of episodes
    for episode in range(EPISODES):
        #see the scores of both players
        player1_scores = [0]
        player2_scores = [0]

        # Reset the environment
        env = PyratEnv.fromPickle(maze_file)
        done = False

        # Generate the states for the 2 players
        # The locations are just reversed
        # The reward received is for player 1, player 2 gets - reward
        state1 = (env.player1_location, env.player2_location, env.cheese_matrix)
        state2 =  (env.player2_location, env.player1_location, env.cheese_matrix)

        i= 0
        while not done:
            decision1 = naive_QLearn1.get_action(state1,env)
            decision2 = naive_QLearn2.get_action(state2,env)
            action = (decision1,decision2)
            #remember the old Q values

            obs, reward, done, _ = env.step(action)

            # make the next states
            next_player1_location = obs['player1_location']
            next_player2_location = obs['player2_location']
            next_cheese_matrix = obs['pieces_of_cheese']
            new_state1 = (next_player1_location, next_player2_location, next_cheese_matrix)
            new_state2 = (next_player2_location, next_player1_location, next_cheese_matrix)

            # Update the q value for player1
            old_value1 = naive_QLearn1.get_q_value(state1, decision1)
            _, next_max = naive_QLearn1.get_max_action(flatten_state(new_state1))
            new_value1 = (1 - naive_QLearn1.alpha) * old_value1 + naive_QLearn1.alpha * (reward + naive_QLearn1.gamma * next_max)
            naive_QLearn1.q_table[flatten_state(state1)][decision1] = new_value1

            # Update the q value for player2
            old_value2 = naive_QLearn2.get_q_value(state2, decision2)
            _, next_max = naive_QLearn2.get_max_action(flatten_state(new_state2))
            new_value2 = (1 - naive_QLearn2.alpha) * old_value2 + naive_QLearn2.alpha * ( (- reward) + naive_QLearn2.gamma * next_max)
            naive_QLearn2.q_table[flatten_state(state2)][decision2] = new_value2

            # add scores to the scores list
            score1, score2 = obs['player_scores']
            player1_scores.append(score1)
            player2_scores.append(score2)

            state1 = new_state1
            state2 = new_state2

            #print(f'turn {i}/2000')
            i+=1

            if done :
                #steps = [i for i in range(len(player2_scores))]
                #plt.plot(steps,player1_scores, 'r')
                #plt.plot(steps,player2_scores, 'b')
                #plt.show()
                print(f"""Done episode {episode} / {EPISODES}
    Player 1 score : {score1}
    Player 2 score : {score2}
    Epsilon value : {naive_QLearn1.epsilon}
""")
                naive_QLearn2.parameter_tune()
                naive_QLearn1.parameter_tune()
    print(f'player 1 q_table : {naive_QLearn1.q_table}/n player2 q_table : {naive_QLearn2.q_table}')

