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

    def test_angle_to(self):
        """
        Test angle_to. Note that (lat, lng) = (y, x)
        """
        lat1, lng1 = 0, 0
        lat2, lng2 = 1, 1
        lat3, lng3 = 1, 0
        lat4, lng4 = 0, -1
        latlng1 = LatLng(lat1, lng1)
        latlng2 = LatLng(lat2, lng2)
        latlng3 = LatLng(lat3, lng3)
        latlng4 = LatLng(lat4, lng4)
        node1 = Node(None, latlng1)
        node2 = Node(None, latlng2)
        node3 = Node(None, latlng3)
        node4 = Node(None, latlng4)
        self.assertEqual(45.0, math.degrees(node1.angle_to(node2)))
        self.assertEqual(90.0, math.degrees(node1.angle_to(node3)))
        self.assertEqual(180.0, math.degrees(node1.angle_to(node4)))

        lat1, lng1 = 38.988152, -76.941595
        lat2, lng2 = 38.988927, -76.940528
        lat3, lng3 = 38.989269, -76.941408
        latlng1 = LatLng(lat1, lng1)
        latlng2 = LatLng(lat2, lng2)
        latlng3 = LatLng(lat3, lng3)
        node1 = Node(None, latlng1)
        node2 = Node(None, latlng2)
        node3 = Node(None, latlng3)
        self.assertAlmostEqual(35.992236322, math.degrees(node1.angle_to(node2)))
        self.assertAlmostEqual(80.4960928229, math.degrees(node1.angle_to(node3)))

if __name__ == '__main__':
    unittest.main()
