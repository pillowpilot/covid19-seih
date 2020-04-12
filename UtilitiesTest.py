import unittest
from Utilities import read_data_file


class ReadDataFileTest(unittest.TestCase):
    def test_usage(self):
        data_file_path = 'testing_resources/cum_world.dat'
        data_table = read_data_file(data_file_path)
        self.assertEqual(len(data_table.columns), 2)

    def test_file_not_found(self):
        random_file_path = 'aFile.abc'
        self.assertRaises(ValueError, read_data_file, random_file_path)


if __name__ == '__main__':
    unittest.main()
