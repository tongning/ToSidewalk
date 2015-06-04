import logging as log
import math
import numpy as np

from latlng import LatLng
from nodes import Node, Nodes
from ways import Sidewalk, Sidewalks
from utilities import window
from network import OSM, parse

log.basicConfig(format="", level=log.DEBUG)


def make_sidewalk_nodes(street, prev_node, curr_node, next_node):
    if prev_node is None:
        v = - curr_node.vector_to(next_node, normalize=False)
        vec_prev = curr_node.vector() + v
        latlng = LatLng(vec_prev[0], vec_prev[1])
        # latlng = LatLng(math.degrees(vec_prev[0]), math.degrees(vec_prev[1]))
        prev_node = Node(None, latlng)
    elif next_node is None:
        v = - curr_node.vector_to(prev_node, normalize=False)
        vec_next = curr_node.vector() + v
        latlng = LatLng(vec_next[0], vec_next[1])
        # latlng = LatLng(math.degrees(vec_next[0]), math.degrees(vec_next[1]))
        next_node = Node(None, latlng)

    curr_latlng = np.array(curr_node.latlng.location())

    v_cp_n = curr_node.vector_to(prev_node)
    v_cn_n = curr_node.vector_to(next_node)
    v_sidewalk = v_cp_n + v_cn_n

    if np.linalg.norm(v_sidewalk) < 0.0000000001:
        v_sidewalk_n = np.array([v_cn_n[1], - v_cn_n[0]])
    else:
        v_sidewalk_n = v_sidewalk / np.linalg.norm(v_sidewalk)

    # The constant is arbitrary.
    # const = 0.000001
    const = street.dist_to_sidewalk
    p1 = curr_latlng + const * v_sidewalk_n
    p2 = curr_latlng - const * v_sidewalk_n
    latlng1 = LatLng(math.degrees(p1[0]), math.degrees(p1[1]))
    latlng2 = LatLng(math.degrees(p2[0]), math.degrees(p2[1]))

    p_sidewalk_1 = Node(None, latlng1)
    p_sidewalk_2 = Node(None, latlng2)

    curr_node.append_sidewalk_node(street.id, p_sidewalk_1)
    curr_node.append_sidewalk_node(street.id, p_sidewalk_2)

    # Figure out on which side you want to put each sidewalk node
    v_c1 = curr_node.vector_to(p_sidewalk_1)
    if np.cross(v_cn_n, v_c1) > 0:
        return p_sidewalk_1, p_sidewalk_2
    else:
        return p_sidewalk_2, p_sidewalk_1


def make_sidewalks(street_network):
    street_nodes, streets = street_network.nodes, street_network.ways
    # Go through each street and create sidewalks on both sides of the road.
    sidewalks = Sidewalks()
    sidewalk_nodes = Nodes()

    for street in streets.get_list():
        sidewalk_1_nodes = []
        sidewalk_2_nodes = []

        # Create sidewalk nodes
        for prev_nid, curr_nid, next_nid in window(street.nids, 3, padding=1):
            curr_node = street_nodes.get(curr_nid)
            prev_node = street_nodes.get(prev_nid)
            next_node = street_nodes.get(next_nid)

            n1, n2 = make_sidewalk_nodes(street, prev_node, curr_node, next_node)
            # log.debug(n1)

            sidewalk_nodes.add(n1.id, n1)
            sidewalk_nodes.add(n2.id, n2)

            sidewalk_1_nodes.append(n1)
            sidewalk_2_nodes.append(n2)
            #log.debug(n1.latlng.location(radian=False))
            #log.debug(n2.latlng.location(radian=False))

        # Keep track of parent-child relationship between streets and sidewalks.
        # And set nodes' adjacency information
        sidewalk_1 = Sidewalk(None, [node.id for node in sidewalk_1_nodes], "footway")
        sidewalk_2 = Sidewalk(None, [node.id for node in sidewalk_2_nodes], "footway")
        sidewalk_1.set_street_id(street.id)
        sidewalk_2.set_street_id(street.id)
        street.append_sidewalk_id(sidewalk_1.id)
        street.append_sidewalk_id(sidewalk_2.id)

        for nid in sidewalk_1.nids:
            curr_node = sidewalk_nodes.get(nid)
            curr_node.append_way(sidewalk_1.id)

        for nid in sidewalk_2.nids:
            curr_node = sidewalk_nodes.get(nid)
            curr_node.append_way(sidewalk_2.id)

        # Add sidewalks to sidewalk_ways
        sidewalks.add(sidewalk_1.id, sidewalk_1)
        sidewalks.add(sidewalk_2.id, sidewalk_2)
    return sidewalk_nodes, sidewalks


def sort_nodes(center_node, nodes):
    """
    Sort nodes around the center_node in clockwise
    """
    def cmp(n1, n2):
        angle1 = (math.degrees(center_node.angle_to(n1)) + 360.) % 360
        angle2 = (math.degrees(center_node.angle_to(n2)) + 360.) % 360

        if angle1 < angle2:
            return -1
        elif angle1 == angle2:
            return 0
        else:
            return 1
    return sorted(nodes, cmp=cmp)


def make_crosswalk_node(node, n1, n2, angle=None):
    v_curr = node.vector()

    if n2 is None and angle is not None:
        v1 = node.vector_to(n1, normalize=True)
        rot_mat = np.array([(math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))])
        v_norm = rot_mat.dot(v1)
        v_new = v_curr + v_norm * node.crosswalk_distance  # const
        latlng_new = LatLng(v_new[0], v_new[1])
    else:
        v1 = node.vector_to(n1, normalize=True)
        v2 = node.vector_to(n2, normalize=True)
        v = v1 + v2
        v_norm = v / np.linalg.norm(v)
        v_new = v_curr + v_norm * node.crosswalk_distance  # const
        latlng_new = LatLng(v_new[0], v_new[1])
    node = Node(None, latlng_new)
    # node.intersection_node_id = node.id
    # node.adjacent_node_ids = [n1.id, n2.id]
    return node


def swap_nodes(sidewalk, nid_from, nid_to):
    index_from = sidewalk.nids.index(nid_from)
    sidewalk.nids[index_from] = nid_to
    return


def create_crosswalk_nodes(intersection_node, adj_street_nodes):
    """
    Create new crosswalk nodes
    :param intersection_node:
    :param adj_street_nodes:
    :return: crosswalk_nodes, source_table
    """
    if len(adj_street_nodes) < 4:
        raise ValueError("You need to pass 4 or more ndoes for adj_street_nodes ")

    source_table = {}
    crosswalk_nodes = []
    for i in range(len(adj_street_nodes)):
        n1 = adj_street_nodes[i - 1]
        n2 = adj_street_nodes[i]
        crosswalk_node = make_crosswalk_node(intersection_node, n1, n2)

        # Keep track of from which streets the crosswalk nodes are created.
        way_ids = []
        for wid in n1.get_way_ids():
            way_ids.append(wid)
        for wid in n2.get_way_ids():
            way_ids.append(wid)
        way_ids = intersection_node.get_shared_way_ids(way_ids)

        crosswalk_node.way_ids = way_ids
        crosswalk_nodes.append(crosswalk_node)
        source_table[crosswalk_node.id] = [intersection_node, n1, n2]

    return crosswalk_nodes, source_table


def connect_crosswalk_nodes(sidewalks, sidewalk_nodes, source_table):
    for crosswalk_node_id in source_table:
        # Get the intersection node and two nodes that created the intersection sidewalk node
        intersection_node, adjacent_street_node1, adjacent_street_node2 = source_table[crosswalk_node_id]
        crosswalk_node = sidewalk_nodes.get(crosswalk_node_id)
        v_crosswalk_node = intersection_node.vector_to(crosswalk_node)  # A vector to the intersection sidewalk node

        # Connect sidewalk nodes created from each street node n1 and n2
        # Get sidewalk nodes that are created from the street node, and
        # identify which one should be connected to crosswalk_node
        for adjacent_street_node in [adjacent_street_node1, adjacent_street_node2]:
            # Create a vector from an intersection node to an adjacent street node
            v_adjacent_street_node = intersection_node.vector_to(adjacent_street_node)

            # Todo: Issue 7
            if len(set(crosswalk_node.get_way_ids())) == 1:
                # The intersection sidewalk node was created from a dummy node and one
                # adjacent street node, so there is only one way_id associated with
                # the intersection sidewalk node (dummy node does not has a way).
                shared_street_id = crosswalk_node.get_way_ids()[0]

                if shared_street_id not in adjacent_street_node.sidewalk_nodes:
                    adjacent_street_node = adjacent_street_node2
            else:
                shared_street_id = intersection_node.get_shared_way_ids(adjacent_street_node)[0]

            sidewalk_node_1_from_adj, sidewalk_node_2_from_adj = adjacent_street_node.sidewalk_nodes[shared_street_id]
            v_adjacent_street_node_s1 = intersection_node.vector_to(sidewalk_node_1_from_adj)

            # Check which one of n1_s1 and n1_s2 are on the same side of the road with crosswalk_node
            # If the rotation (cross product) from v_n1 to v_crosswalk_node is same as v_n1 to v_n1_s1, then
            # n1_s1 should be on the same side. Otherwise, n1_s1 should be on the same side with crosswalk_node.
            if np.cross(v_adjacent_street_node, v_crosswalk_node) * np.cross(v_adjacent_street_node, v_adjacent_street_node_s1) > 0:
                sidewalk_node_from_adj = sidewalk_node_1_from_adj
            else:
                sidewalk_node_from_adj = sidewalk_node_2_from_adj

            # Identify on which sidewalk adjacent_street_node_sidewalk_temp belongs too.
            sidewalk_id = sidewalk_node_from_adj.get_way_ids()[0]
            sidewalk = sidewalks.get(sidewalk_id)

            # Swap the sidewalk intersection_sidewalk_node with crosswalk_node
            potential_nodes_to_swap = [n.id for n in intersection_node.get_sidewalk_nodes(shared_street_id)]
            intersection_sidewalk_node_ids = set(sidewalk.nids) & set(potential_nodes_to_swap)
            intersection_sidewalk_node_id = list(intersection_sidewalk_node_ids)[0]
            sidewalk.swap_nodes(intersection_sidewalk_node_id, crosswalk_node.id)
            sidewalk_nodes.remove(intersection_sidewalk_node_id)

            if len(set(crosswalk_node.get_way_ids())) == 1:
                break

    return sidewalk_nodes, sidewalks


def make_crosswalks(street_network, sidewalk_network):
    # Some helper functions
    street_nodes, streets = street_network.nodes, street_network.ways
    sidewalk_nodes, sidewalks = sidewalk_network.nodes, sidewalk_network.ways

    intersection_node_ids = list(streets.intersection_node_ids)
    intersection_nodes = [street_nodes.get(nid) for nid in intersection_node_ids]

    # Create sidewalk nodes for each intersection node and overwrite the adjacency information
    for intersection_node in intersection_nodes:
        adj_street_nodes = street_network.get_adjacent_nodes(intersection_node)
        adj_street_nodes = sort_nodes(intersection_node, adj_street_nodes)
        v_curr = intersection_node.vector()

        if len(adj_street_nodes) == 3:
            # Take care of the case where len(adj_nodes) == 3.
            # Identify the largest angle that are formed by three segments
            # Make a dummy node between two vectors that form the largest angle
            # Using the four nodes (3 original nodes and a dummy node), create crosswalk nodes
            vectors = [intersection_node.vector_to(adj_street_node, normalize=True) for adj_street_node in adj_street_nodes]
            angles = [math.acos(np.dot(vectors[i - 1], vectors[i])) for i in range(3)]

            idx = np.argmax(angles)
            vec_idx = (idx + 1) % 3
            dummy_vector = - vectors[vec_idx] * 0.0001
            dummy_coordinate_vector = v_curr + dummy_vector
            dummy_latlng = LatLng(dummy_coordinate_vector[0], dummy_coordinate_vector[1])
            dummy_node = Node(None, dummy_latlng)
            adj_street_nodes.insert(idx, dummy_node)

        # Create crosswalk nodes and add a cross walk to the data structure
        crosswalk_nodes, source_table = create_crosswalk_nodes(intersection_node, adj_street_nodes)
        crosswalk_node_ids = [node.id for node in crosswalk_nodes]
        crosswalk_node_ids.append(crosswalk_node_ids[0])
        crosswalk = Sidewalk(None, crosswalk_node_ids, "crosswalk")
        for crosswalk_node in crosswalk_nodes:
            sidewalk_nodes.add(crosswalk_node.id, crosswalk_node)
            sidewalk_nodes.crosswalk_node_ids.append(crosswalk_node.id)

        sidewalks.add(crosswalk.id, crosswalk)

        # Connect the crosswalk nodes with correct sidewalk nodes
        sidewalk_nodes, sidewalks = connect_crosswalk_nodes(sidewalks, sidewalk_nodes, source_table)
    return sidewalk_nodes, sidewalks


def main(street_network):
    street_nodes = street_network.nodes
    streets = street_network.ways
    sidewalk_nodes, sidewalks = make_sidewalks(street_network)
    sidewalk_network = OSM(sidewalk_nodes, sidewalks)
    make_crosswalks(street_network, sidewalk_network)

    output = sidewalk_network.export(format='geojson')
    return output


if __name__ == "__main__":
    filename = "../resources/SmallMap_01.osm"
    # filename = "../resources/ParallelLanes_01.osm"
    nodes, ways = parse(filename)
    street_network = OSM(nodes, ways)
    street_network.parse_intersections()

    print main(street_network)
