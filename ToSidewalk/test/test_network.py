import unittest
from ToSidewalk.network import *
from ToSidewalk.nodes import *
from ToSidewalk.ways import *


class TestNetworkMethods(unittest.TestCase):
    def test_segment_parallel_streets(self):
        """
        Test segment parallel streets
        """
        segment1_coordinates = [(float(i + 1), float(i)) for i in range(0, 7)]
        segment2_coordinates = [(float(i + 1), float(i)) for i in range(0, 7)]

        nodes = Nodes()
        segment1_nodes = [Node('s1_' + str(i), coord[1], coord[0]) for i, coord in enumerate(segment1_coordinates)]
        segment2_nodes = [Node('s2_' + str(i), coord[1], coord[0]) for i, coord in enumerate(segment2_coordinates)]
        # answer_segment_nodes = [Node(None, LatLng(coord[1], coord[0])) for coord in answer_segment_coordinates]

        for node in segment1_nodes:
            nodes.add(node)
        for node in segment2_nodes:
            nodes.add(node)

        segment1_node_ids = [node.id for node in segment1_nodes]
        segment2_node_ids = [node.id for node in segment2_nodes]
        # answer_segment_node_ids = [node.id for node in answer_segment_nodes]

        street1 = Street(1, segment1_node_ids)
        street2 = Street(2, segment2_node_ids)
        # answer_street = Street(None, answer_segment_node_ids)

        streets = Streets()
        streets.add(street1)
        streets.add(street2)

        network = OSM(nodes, streets, None)

        overlap, segments1, segments2 = network.segment_parallel_streets([street1, street2])

        self.assertEqual(segments1[0], [])
        self.assertEqual(segments1[2], [])
        self.assertListEqual(segments1[1], segment1_node_ids)
        self.assertEqual(segments2[0], [])
        self.assertEqual(segments2[2], [])
        self.assertListEqual(segments2[1], segment2_node_ids)

    def test_segment_parallel_streets2(self):
        """
        Test segment parallel streets
        """
        segment1_coordinates = [(float(i + 1), float(i)) for i in range(2, 10)]
        segment2_coordinates = [(float(i + 1), float(i)) for i in range(0, 7)]
        answer_segment_coordinates = [(float(2 * i + 1) / 2, float(2 * i + 1) / 2) for i in range(0, 7)]

        nodes = Nodes()
        segment1_nodes = [Node('s1_' + str(i), coord[1], coord[0]) for i, coord in enumerate(segment1_coordinates)]
        segment2_nodes = [Node('s2_' + str(i), coord[1], coord[0]) for i, coord in enumerate(segment2_coordinates)]
        # answer_segment_nodes = [Node(None, LatLng(coord[1], coord[0])) for coord in answer_segment_coordinates]

        for node in segment1_nodes:
            nodes.add(node)
        for node in segment2_nodes:
            nodes.add(node)

        segment1_node_ids = [node.id for node in segment1_nodes]
        segment2_node_ids = [node.id for node in segment2_nodes]
        # answer_segment_node_ids = [node.id for node in answer_segment_nodes]

        street1 = Street(1, segment1_node_ids)
        street2 = Street(2, segment2_node_ids)
        # answer_street = Street(None, answer_segment_node_ids)

        streets = Streets()
        streets.add(street1)
        streets.add(street2)

        network = OSM(nodes, streets, None)

        overlap, segments1, segments2 = network.segment_parallel_streets([street1, street2])

        self.assertEqual(segments1[0], [])
        self.assertEqual(segments1[2], segment1_node_ids[-3:])
        self.assertListEqual(segments1[1], segment1_node_ids[:-2])
        self.assertEqual(segments2[0], segment2_node_ids[:2])
        self.assertEqual(segments2[2], [])
        self.assertListEqual(segments2[1], segment2_node_ids[1:])

    def test_merge_parallel_street_segments(self):
        """
        Test the constructor
        """
        segment1_coordinates = [(float(i + 1), float(i)) for i in range(0, 7)]
        segment2_coordinates = [(float(i + 1), float(i)) for i in range(0, 7)]
        answer_segment_coordinates = [(float(2 * i + 1) / 2, float(2 * i + 1) / 2) for i in range(0, 7)]

        nodes = Nodes()
        segment1_nodes = [Node(None, coord[1], coord[0]) for coord in segment1_coordinates]
        segment2_nodes = [Node(None, coord[1], coord[0]) for coord in segment2_coordinates]
        answer_segment_nodes = [Node(None, coord[1], coord[0]) for coord in answer_segment_coordinates]

        for node in segment1_nodes:
            nodes.add(node)
        for node in segment2_nodes:
            nodes.add(node)

        segment1_node_ids = [node.id for node in segment1_nodes]
        segment2_node_ids = [node.id for node in segment2_nodes]
        answer_segment_node_ids = [node.id for node in answer_segment_nodes]

        street1 = Street(1, segment1_node_ids)
        street2 = Street(2, segment2_node_ids)
        answer_street = Street(None, answer_segment_node_ids)

        streets = Streets()
        streets.add(street1)
        streets.add(street2)

        network = OSM(nodes, streets, None)

        merged_segment = network.merge_parallel_street_segments([(street1.id, street2.id)])

        # self.assertEqual(answer_segment_coordinates[0], merged_segment[0])

    def test_simplify(self):
        segment1_coordinates = [
            (0., 2.),
            (3., 0.),
            (6., 1.),
            (9., 0.),
            (12., 0.5),
            (15., 2.)
        ]

        nodes = Nodes()
        segment1_nodes = [Node('s1_' + str(i), coord[1], coord[0]) for i, coord in enumerate(segment1_coordinates)]
        for node in segment1_nodes:
            nodes.add(node)

        segment1_node_ids = [node.id for node in segment1_nodes]
        street1 = Street(1, segment1_node_ids)
        streets = Streets()
        streets.add(street1)

        network = OSM(nodes, streets, None)
        network.simplify(street1.id)

    def test_parse(self):
        filename = "../../resources/SmallMap_01.osm"
        nodes, ways, bounds = parse(filename)

    def test_split_streets(self):
        filename = "../../resources/SmallMap_01.osm"
        nodes, ways, bounds = parse(filename)
        street_network = OSM(nodes, ways, bounds)
        street_network.preprocess()

if __name__ == '__main__':
    unittest.main()


