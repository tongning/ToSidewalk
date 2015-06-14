import unittest
from ToSidewalk.nodes import *
from ToSidewalk.latlng import *
from ToSidewalk.ways import *
from ToSidewalk.network import *

class TestNodeMethods(unittest.TestCase):
    def test_constructor(self):
        """
        Test the constructor
        """
        node = Node(None, 0, 0)
        self.assertNotEqual(str(0), node.id)

        node = Node(0, 0, 0)
        self.assertEqual(str(0), node.id)

        lat, lng = 38.898556, -77.037852
        node = Node(None, lat, lng)
        self.assertEqual(node.lat, lat)
        self.assertEqual(node.lng, lng)

    def test_angle_to(self):
        """
        Test angle_to. Note that (lat, lng) = (y, x)
        """
        lat1, lng1 = 0, 0
        lat2, lng2 = 1, 1
        lat3, lng3 = 1, 0
        lat4, lng4 = 0, -1
        node1 = Node(None, lat1, lng1)
        node2 = Node(None, lat2, lng2)
        node3 = Node(None, lat3, lng3)
        node4 = Node(None, lat4, lng4)
        self.assertEqual(45.0, math.degrees(node1.angle_to(node2)))
        self.assertEqual(90.0, math.degrees(node1.angle_to(node3)))
        self.assertEqual(180.0, math.degrees(node1.angle_to(node4)))

        lat1, lng1 = 38.988152, -76.941595
        lat2, lng2 = 38.988927, -76.940528
        lat3, lng3 = 38.989269, -76.941408
        node1 = Node(None, lat1, lng1)
        node2 = Node(None, lat2, lng2)
        node3 = Node(None, lat3, lng3)
        self.assertAlmostEqual(35.992236322, math.degrees(node1.angle_to(node2)))
        self.assertAlmostEqual(80.4960928229, math.degrees(node1.angle_to(node3)))

    def test_belongs_to(self):
        node = Node(None, 0, 0)
        nodes = Nodes()
        nodes.add(node)
        self.assertEqual(node.belongs_to(), nodes)

    def test_vector(self):
        node1 = Node(None, 0, 0)
        v = node1.vector()
        self.assertEqual(v[0], 0)
        self.assertEqual(v[1], 0)

    def test_vector_to(self):
        node1 = Node(None, 0, 0)
        node2 = Node(None, 1, 1)
        v = node1.vector_to(node2)
        self.assertEqual(v[0], 1.)
        self.assertEqual(v[1], 1.)

        v = node1.vector_to(node2, normalize=True)
        self.assertEqual(v[0], 1 / math.sqrt(2))
        self.assertEqual(v[1], 1 / math.sqrt(2))

        node3 = Node(None, 1, 1)
        v = node2.vector_to(node3)
        self.assertEqual(v[0], 0)
        self.assertEqual(v[1], 0)

class TestNodesMethods(unittest.TestCase):
    def test_belongs_to(self):
        """
        Test the constructor
        """
        ways = Ways()
        nodes = Nodes()
        network = Network(nodes, ways)
        self.assertEqual(nodes.belongs_to(), network)


if __name__ == '__main__':
    unittest.main()
