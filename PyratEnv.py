# Imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from imports.maze import *
from imports.display import *
from pyrat import move


class PyratEnv(gym.Env):
    # TODO : add mud
    """
    Pyrat Maze environment
    For now without mud for simplification
    Inputs :
        width : the width of the maze
        height : the height of the maze
        nb_pieces_of_cheese : the number of pieces of cheese in the maze
        max_turns : max number of turns to avoid infinite games
    """
    metadata = {'render.modes': ['human', 'none']}

    def __init__(self, width=21, height=15, nb_pieces_of_cheese=41, max_turns=2000):
        self.max_turns = max_turns
        self.turn = 0
        self.maze = None
        self.maze_dimension = (width,height)
        self.pieces_of_cheese = []
        self.player1_location = None
        self.player2_location = None
        self.player1_score = 0
        self.player2_score = 0
        self.player1_misses = 0
        self.player2_misses = 0
        self.player1_moves = 0
        self.player2_moves = 0
        self.random_seed = random.randint(0, sys.maxsize)
        self.width, self.height, self.pieces_of_cheese, self.maze = generate_maze(width, height, 0.7,
                                                                                  True, True,
                                                                                  0.7, 10, "",
                                                                                  self.random_seed)
        self.pieces_of_cheese, self.player1_location, self.player2_location = generate_pieces_of_cheese(nb_pieces_of_cheese, width, height,
                                                                                         True,
                                                                                         self.player1_location,
                                                                                         self.player2_location,
                                                                                         False)

    def step(self, action):
        (decision1, decision2) = action
        self.turn+=1
        # Perform both player's actions on the maze variables
        self.player1_location,self.player2_location, stuck1, stuck2, self.player1_moves,self.player2_moves, self.player1_misses,self.player2_misses = move(decision1,decision2,self.maze, self.player1_location,self.player2_location,0,0,self.player1_moves,self.player2_moves,self.player1_misses,self.player2_misses)
        difference = self.player1_score - self. player2_score

        # Calculate the return variables
        observation = [self.maze,self.pieces_of_cheese,self.turn, self.player1_score,self.player2_score,self.player1_location,self.player2location]
        reward = difference
        done = self.turn >= self.max_turns
        info = dict()

        return observation,reward,done,info

    def reset(self):
        #reset the maze randomly
        self.random_seed = random.randint(0, sys.maxsize)
        self.turn = 0
        self.pieces_of_cheese = []
        self.width, self.height, self.pieces_of_cheese, self.maze = generate_maze(self.maze_dimension[0], self.maze_dimension[1], 0.7,
                                                                                  True, True,
                                                                                  0.7, 10, "",
                                                                                  self.random_seed)
        self.pieces_of_cheese, self.player1_location, self.player2_location = generate_pieces_of_cheese(args.pieces,
                                                                                                        self.maze_dimension[0], self.maze_dimension[1],
                                                                                                        True,
                                                                                                        self.player1_location,
                                                                                                        self.player2_location,
                                                                                                        False)
        # Reset player turns, score, misses
        self.player1_score, self.player2_score,self.player1_misses,self.player2_misses,self.player1_moves,self.player2_moves = 0,0,0,0,0,0

    def render(self, mode='human'):
        pass

if __name__ == '__main__':
    maze  = PyratEnv()
    print(maze.maze)