import unittest
import staty


class StatyTest(unittest.TestCase):

    def test_mean(self):
        self.assertEqual(staty.mean([2, 4, 6, 8]), 5)
        with self.assertRaises(ValueError):
            staty.mean([2])

    def test_variance(self):
        self.assertEqual(staty.var([2, 4, 6, 8]), 6.666666666666667)
        with self.assertRaises(ValueError):
            staty.var([2])

    def test_stdev(self):
        self.assertAlmostEqual(staty.stdev([2, 4, 6, 8]), 2.581988897471611)
        with self.assertRaises(ValueError):
            staty.stdev([2])

    def test_stderr(self):
        self.assertAlmostEqual(staty.stderr([1, 2, 3, 4]), 0.6454972243679028)
        with self.assertRaises(ValueError):
            staty.stdev([2])

    def test_median(self):
        self.assertEqual(staty.median([2, 4, 6, 8, 10]), 6)
        self.assertEqual(staty.median([2, 4, 6, 8]), 5)
        self.assertEqual(staty.median(['a', 'b', 'c', 'd']), ('b', 'c'))
        with self.assertRaises(ValueError):
            staty.median([2])

    def test_mode(self):
        self.assertEqual(staty.mode([2, 2, 3, 4, 5, 6]), 2)
        self.assertEqual(staty.mode([2, 3, 4, 5, 6]), [2, 3, 4, 5, 6])
        with self.assertRaises(ValueError):
            staty.mode([2])

    def test_cv(self):
        self.assertAlmostEqual(staty.cv([2, 4, 6, 8]), 0.5163977794943222)
        with self.assertRaises(ValueError):
            staty.cv([2])

    def test_cov(self):
        self.assertAlmostEqual(staty.cov([1, 2, 3, 4], [5, 6, 7, 8]), 1.6666666666666667)
        with self.assertRaises(ValueError):
            staty.cov([1], [1])

    def test_correlation_r(self):
        self.assertAlmostEqual(staty.correlation_r([1, 2, 3, 4], [5, 6, 7, 8]), 1.0)
        with self.assertRaises(ValueError):
            staty.correlation_r([1], [1])

    def test_zscore(self):
        self.assertEqual(staty.zscore([1, 2, 3, 4]),
                         [-1.161895003862225, -0.3872983346207417, 0.3872983346207417, 1.161895003862225])
        with self.assertRaises(ValueError):
            staty.zscore([1])


if __name__ == '__main__':
    unittest.main()
