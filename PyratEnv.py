# Imports
import time

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
    action_space = spaces.Discrete(4)

    def __init__(self, width=21, height=15, nb_pieces_of_cheese=41, max_turns=2000,no_mud = True ):
        self.max_turns = max_turns
        self.turn = 0
        self.maze = None
        self.maze_dimension = (width, height)
        self.pieces_of_cheese = []
        self.nb_pieces_of_cheese = nb_pieces_of_cheese
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
        self.pieces_of_cheese, self.player1_location, self.player2_location = generate_pieces_of_cheese(
            nb_pieces_of_cheese, width, height,
            True,
            self.player1_location,
            self.player2_location,
            False)
        if no_mud:
            for start in self.maze :
                for end in self.maze[start]:
                    self.maze[start][end] = 1

    def _calculate_reward(self):
        """
        Returns the reward for the current turn and removes the potential cheeses taht have been eaten
        reward is 1 if the player 1 eats a piece of cheese and player 2 doesnt, -1 if player 2 does and player 1 doesnt and 0 in all other cases
        :return: reward
        """
        reward = 0
        if self.player1_location in self.pieces_of_cheese:
            self.pieces_of_cheese.remove(self.player1_location)
            if self.player2_location == self.player1_location:
                self.player2_score += 0.5
                self.player1_score += 0.5
                # reward = 0
            else:
                self.player1_score += 1
                reward = 1
        if self.player2_location in self.pieces_of_cheese:
            self.pieces_of_cheese.remove(self.player2_location)
            self.player2_score += 1
            reward = -1
        return reward

    def _move(self, action):
        """
        imports.maze.move function wrapper
        :param action: (decision1,decision2) of both players
        """
        (decision1, decision2) = action
        self.player1_location, self.player2_location, stuck1, stuck2, self.player1_moves, self.player2_moves, self.player1_misses, self.player2_misses = move(
            decision1, decision2, self.maze, self.player1_location, self.player2_location, 0, 0, self.player1_moves,
            self.player2_moves, self.player1_misses, self.player2_misses)

    def _check_done(self):
        return (self.turn >= self.max_turns) or (self.player1_score > (self.nb_pieces_of_cheese) / 2) or (
                self.player2_score > (self.nb_pieces_of_cheese) / 2)

    def step(self, action):
        self.turn += 1
        # Perform both player's actions on the maze variables
        self._move()

        reward = self._calculate_reward()

        # Calculate the return variables
        observation = [self.maze, self.pieces_of_cheese, self.turn, self.player1_score, self.player2_score,
                       self.player1_location, self.player2location]
        done = self._check_done()
        info = dict()

        return observation, reward, done, info

    def reset(self):
        # reset the maze randomly
        self.random_seed = random.randint(0, sys.maxsize)
        self.turn = 0
        self.pieces_of_cheese = []
        self.width, self.height, self.pieces_of_cheese, self.maze = generate_maze(self.maze_dimension[0],
                                                                                  self.maze_dimension[1], 0.7,
                                                                                  True, True,
                                                                                  0.7, 10, "",
                                                                                  self.random_seed)
        self.pieces_of_cheese, self.player1_location, self.player2_location = generate_pieces_of_cheese(args.pieces,
                                                                                                        self.maze_dimension[
                                                                                                            0],
                                                                                                        self.maze_dimension[
                                                                                                            1],
                                                                                                        True,
                                                                                                        self.player1_location,
                                                                                                        self.player2_location,
                                                                                                        False)
        # Reset player turns, score, misses
        self.player1_score, self.player2_score, self.player1_misses, self.player2_misses, self.player1_moves, self.player2_moves = 0, 0, 0, 0, 0, 0

    # TODO : Rendering
    # TODO : Sound
    def render(self, mode='human'):
        if mode == 'human':
            pass
        elif mode == "none":
            pass


if __name__ == '__main__':
    maze = PyratEnv()
    import pygame
    print(maze.maze, maze.pieces_of_cheese)
    pygame.init()
    screen = pygame.display.set_mode((1920,1080))

    screen.fill((255,255,255))
    print(screen)
    window_width, window_height = 1920,1080
    scale, offset_x, offset_y, image_background, image_cheese, image_corner, image_moving_python, image_moving_rat, image_python, image_rat, image_wall, image_mud, image_portrait_python, image_portrait_rat, tiles, image_tile = init_coords_and_images(
        maze.width, maze.height, True, True, window_width, window_height)


    bg = build_background(screen, maze.maze, tiles, image_background, image_tile, image_wall, image_corner, image_mud, offset_x, offset_y, maze.width, maze.height, window_width, window_height, image_portrait_rat, image_portrait_python, scale, True, True),(0,0)
    print(bg)
    screen.blit(bg[0],(0,0))
    running = True
    while running:
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False