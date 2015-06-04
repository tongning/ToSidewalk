import unittest
import random
from ToSidewalk.ToSidewalk import *

class TestToSidewalkMethods(unittest.TestCase):
    def test_sort_nodes(self):
        """
        """
        center = "38.988152,-76.941595"
        center = map(float, center.split(","))
        center_node = Node('0', LatLng(center[0], center[1]))

        latlngs = """
38.988927,-76.940528
38.989269,-76.941408
38.988239,-76.942878
38.987172,-76.942540
38.987105,-76.941494
38.987906,-76.940222
        """

        latlngs = latlngs.strip().split('\n')
        latlngs = [map(float, latlng.split(',')) for latlng in latlngs]

        nodes1 = [Node(str(i), LatLng(latlng[0], latlng[1])) for i, latlng in enumerate(latlngs)]
        nodes2 = [Node(str(i), LatLng(latlng[0], latlng[1])) for i, latlng in enumerate(latlngs)]

        random.shuffle(nodes1)
        nodes1 = sort_nodes(center_node, nodes1)

        for node1, node2 in zip(nodes1, nodes2):
            self.assertTrue(node1.latlng == node2.latlng)

    def test_make_crosswalk_node(self):
        clat, clng = 0, 0
        lat1, lng1 = 0, 1
        lat2, lng2 = 1, 0
        cnode = Node('0', LatLng(clat, clng))
        node1 = Node('1', LatLng(lat1, lng1))
        node2 = Node('2', LatLng(lat2, lng2))

        rlat, rlng = cnode.crosswalk_distance / math.sqrt(2), cnode.crosswalk_distance / math.sqrt(2)
        node = make_crosswalk_node(cnode, node1, node2)
        self.assertEqual(rlat, node.latlng.lat)
        self.assertEqual(rlng, node.latlng.lng)

    def test_create_crosswalk_nodes(self):
        clat, clng = 0, 0
        lat1, lng1 = 0, 1
        lat2, lng2 = 1, 0
        lat3, lng3 = 0, -1
        lat4, lng4 = -1, 0
        cnode = Node('0', LatLng(clat, clng))
        node1 = Node('1', LatLng(lat1, lng1))
        node2 = Node('2', LatLng(lat2, lng2))
        node3 = Node('3', LatLng(lat3, lng3))
        node4 = Node('4', LatLng(lat4, lng4))

        # Parameter Error:
        # http://stackoverflow.com/questions/256222/which-exception-should-i-raise-on-bad-illegal-argument-combinations-in-python
        adjacent_nodes = [node1, node2]
        self.assertRaises(ValueError, create_crosswalk_nodes, cnode, adjacent_nodes)
        adjacent_nodes = [node1, node2, node3]
        self.assertRaises(ValueError, create_crosswalk_nodes, cnode, adjacent_nodes)

        # Opposite of assertRaises
        # http://stackoverflow.com/questions/4319825/python-unittest-opposite-of-assertraises
        try:
            adjacent_nodes = [node1, node2, node3, node4]
            create_crosswalk_nodes(cnode, adjacent_nodes)
        except ValueError:
            self.fail("create_crosswalk_nodes() failed unexpectedly")

    def test_swap_nodes(self):
        pass

    def test_main(self):
        filenames = [
            "../../resources/Simple4WayIntersection_01.osm",
            "../../resources/Simple4WayIntersection_02.osm",
            "../../resources/TShapeIntersection_01.osm",
            "../../resources/TShapeIntersection_02.osm",
            "../../resources/SegmentedStreet_01.osm",
            "../../resources/ComplexIntersection_01.osm",
            "../../resources/SmallMap_01.osm",
            "../../resources/SmallMap_02.osm"
            # "../../resources/ParallelLanes_01.osm"  # Todo. This fails.
        ]
        for filename in filenames:
            nodes, ways = parse(filename)
            osm_obj = OSM(nodes, ways)
            osm_obj.parse_intersections()

            # output = osm_obj.export()
            # print output
            main(osm_obj)

if __name__ == '__main__':
    unittest.main()
