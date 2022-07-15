import unittest
import numpy as np
from utils import transform_votes_powerset


class TestUtils(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_transform_votes_powerset(self):
        votes = np.array([
            [[0, 0, 0, 1], [0, 1, 1, 1]],
            [[0, 0, 0, 1], [1, 1, 1, 0]]
        ])
        votes, map_idx_label = transform_votes_powerset(
            input_votes=votes)
        print('votes: ', votes)
        print('map_idx_label: ', map_idx_label)
        np.testing.assert_equal(votes[0], [0, 1, 0, 0, 0, 0, 0, 1,
                                           0, 0, 0, 0, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
