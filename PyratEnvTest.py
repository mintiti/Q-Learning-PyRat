import unittest
from pyratEnv.PyratEnv import PyratEnv

class TestPyratEnvInit(unittest.TestCase):
    def test_defaut_init(self):
        env = PyratEnv()
        self.assertEqual(env.max_turns, 2000)
        self.assertEqual(env.turn, 0)
        self.assertEqual(env.maze_dimension, (21,15))
        self.assertEqual(env.nb_pieces_of_cheese, 41)
        self.assertEqual(env.player1_location, (0,0))
        self.assertEqual(env.player2_location, (20,14))
        self.assertEqual(env.player1_score, 0)
        self.assertEqual(env.player2_score, 0)
        self.assertEqual(env.player1_misses,0)
        self.assertEqual(env.player2_misses,0)
        self.assertEqual(env.player1_moves, 0)
        self.assertEqual(env.player2_moves, 0)
        self.assertIsInstance(env.random_seed, int)

if __name__ == '__main__':
    unittest.main()
