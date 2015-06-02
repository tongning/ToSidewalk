import unittest
from ToSidewalk.nodes import *
from ToSidewalk.latlng import *

class TestNodeMethods(unittest.TestCase):
    def test_constructor(self):
        """
        Test the constructor
        """
        node = Node()
        self.assertNotEqual(str(0), node.id)

        node = Node(0)
        self.assertEqual(str(0), node.id)

        lat, lng = 38.898556, -77.037852
        latlng = LatLng(lat, lng)
        node = Node(None, latlng=latlng)
        self.assertEqual(latlng, node.latlng)
        self.assertEqual(latlng.lat, lat)
        self.assertEqual(latlng.lng, lng)


if __name__ == '__main__':
    unittest.main()
