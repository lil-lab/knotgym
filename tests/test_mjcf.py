import unittest
from knotgym.mjcf import split_list


class TestSplit(unittest.TestCase):
  def test_simple(self):
    actual = split_list([1, 2, 3, 4, 5], 3, create_overlap=False)
    expected = [[1, 2], [3, 4], [5]]
    self.assertEqual(actual, expected)

  def test_simple_overlap(self):
    actual = split_list([1, 2, 3, 4, 5], 3, create_overlap=True)
    expected = [[1, 2, 3], [3, 4, 5], [5, 1]]
    self.assertEqual(actual, expected)

  def test_large_n(self):
    actual = split_list([1, 2, 3, 4, 5], 5, create_overlap=False)
    expected = [[1], [2], [3], [4], [5]]
    self.assertEqual(actual, expected)

  def test_large_n_overlap(self):
    actual = split_list([1, 2, 3, 4, 5], 5, create_overlap=True)
    expected = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1]]
    self.assertEqual(actual, expected)
