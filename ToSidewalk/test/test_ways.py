import unittest
from ToSidewalk.ways import *

class TestWayMethods(unittest.TestCase):
    def test_constructor(self):
        """
        Test the constructor
        """
        way = Way()
        self.assertNotEqual(way.id, '0')

        way = Way(0)
        self.assertEqual(way.id, '0')

        nids = (1, 2, 3)
        way = Way(0, (1, 2, 3))
        self.assertTrue(way.get_node_ids()[0], nids[0])
        self.assertTrue(way.get_node_ids()[1], nids[1])
        self.assertTrue(way.get_node_ids()[2], nids[2])


if __name__ == '__main__':
    unittest.main()
