import tensorflow as tf
import PyratEnv
import random as rd
import numpy as np

class DeepQLearning :
    def __init__(self,  env,alpha = 0.99, gamma = 0.8, epsilon = 0.99):
        self.nn_prediction = None
        self.nn_target = None
        self.nb_episode = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.maze_dimension = env.maze_dimension
        self.env =env

        # Initialize the NNs

        self.initialize_prediction_nn()
        self.initialize_target_nn()

    def initialize_target_nn(self):
        nb_parameters_matrix = self.maze_dimension[0] * self.maze_dimension[1]
        self.nn_target = tf.keras.Sequential()
        self.nn_target.add(tf.keras.layers.Dense(128, input_dim =nb_parameters_matrix + 7, activation ='relu'))
        self.nn_target.add(tf.keras.layers.Dense(64, activation ='relu'))
        self.nn_target.add(tf.keras.layers.Dense(32, activation='relu'))
        self.nn_target.add(tf.keras.layers.Dense(4))


        self.nn_target.compile(optimizer ='adam', loss ='mse')

    def initialize_prediction_nn(self):
        nb_parameters_matrix = self.maze_dimension[0] * self.maze_dimension[1]
        self.nn_prediction = tf.keras.Sequential()
        self.nn_prediction.add(tf.keras.layers.Dense(128, input_dim =nb_parameters_matrix + 7, activation ='relu'))
        self.nn_prediction.add(tf.keras.layers.Dense(64, activation ='relu'))
        self.nn_prediction.add(tf.keras.layers.Dense(32, activation='relu'))
        self.nn_prediction.add(tf.keras.layers.Dense(4))


        self.nn_prediction.compile(optimizer ='adam', loss ='mse')


    def update_target_nn(self):
        self.nn_target.set_weights(self.nn_prediction.get_weights())

    def train_prediction(self,inputs, targets,):
        self.nn_prediction.fit(x= inputs, y = targets, batch_size = 32)

    def act(self,state):
        if rd.uniform(0,1) <= self.epsilon :
            return self.env.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values)


if __name__ == '__main__':
    pass