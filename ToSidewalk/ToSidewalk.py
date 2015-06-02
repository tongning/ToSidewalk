import logging as log
import math
import numpy as np

from latlng import LatLng
from nodes import Node, Nodes
from ways import Sidewalk, Sidewalks
from utilities import window
from osm import OSM, parse

log.basicConfig(format="", level=log.DEBUG)


def make_sidewalk_nodes(street, prev_node, curr_node, next_node):
    if prev_node is None:
        v = - curr_node.vector_to(next_node, normalize=False)
        vec_prev = curr_node.vector() + v
        latlng = LatLng(math.degrees(vec_prev[0]), math.degrees(vec_prev[1]))
        prev_node = Node(None, latlng)
    elif next_node is None:
        v = - curr_node.vector_to(prev_node, normalize=False)
        vec_next = curr_node.vector() + v
        latlng = LatLng(math.degrees(vec_next[0]), math.degrees(vec_next[1]))
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


def make_sidewalks(street_nodes, streets):
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
    const = 0.000001414
    v_curr = node.vector()

    if n2 is None and angle is not None:
        v1 = node.vector_to(n1, normalize=True)
        rot_mat = np.array([(math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))])
        v_norm = rot_mat.dot(v1)
        v_new = v_curr + v_norm * const
        latlng_new = LatLng(math.degrees(v_new[0]), math.degrees(v_new[1]))
        return Node(None, latlng_new)
    else:
        v1 = node.vector_to(n1, normalize=True)
        v2 = node.vector_to(n2, normalize=True)
        v = v1 + v2
        v_norm = v / np.linalg.norm(v)
        v_new = v_curr + v_norm * const
        latlng_new = LatLng(v_new[0], v_new[1])
        return Node(None, latlng_new)


def swap_nodes(sidewalk_nodes, sidewalk, nid_from, nid_to):
    index_from = sidewalk.nids.index(nid_from)
    sidewalk.nids[index_from] = nid_to
    sidewalk_nodes.remove(nid_from)
    return sidewalk_nodes, sidewalk


def make_crosswalks(street_nodes, sidewalk_nodes, streets, sidewalks):
    # Some helper functions

    intersection_node_ids = streets.intersection_node_ids
    intersection_nodes = [street_nodes.get(nid) for nid in intersection_node_ids]

    # Create sidewalk nodes for each intersection node and overwrite the adjacency information
    for intersection_node in intersection_nodes:
        street_ids = intersection_node.get_way_ids()
        adj_street_nodes = []
        for street_id in street_ids:
            street = streets.get(street_id)

            # If the current intersection node is at the head of street.nids, then take the second node and push it
            # into adj_street_nodes. Otherwise, take the node that is second to the last in street.nids .
            if street.nids[0] == intersection_node.id:
                adj_street_nodes.append(street_nodes.get(street.nids[1]))
            else:
                adj_street_nodes.append(street_nodes.get(street.nids[-2]))

        adj_street_nodes = sort_nodes(intersection_node, adj_street_nodes)
        v_curr = intersection_node.vector()

        # Creat new intersection sidewalk nodes
        # Record from which street nodes each intersection node is created with source_table
        source_table = {}
        new_crosswalk_node_ids = []
        if len(adj_street_nodes) == 3:
            # Take care of the case where len(adj_nodes) == 3.
            # Identify the largest angle that are formed by three segments
            # Make a dummy node between two vectors that form the largest angle
            # Using the four nodes (3 original nodes and a dummy node), create crosswalk nodes
            vectors = [intersection_node.vector_to(adj_street_node, normalize=True) for adj_street_node in adj_street_nodes]
            angles = [math.acos(np.dot(vectors[i - 1], vectors[i])) for i in range(3)]

            idx = np.argmax(angles)
            vec_idx = (idx + 1) % 3
            dummy_vector = - vectors[vec_idx] * 0.000001
            dummy_coordinate_vector = v_curr + dummy_vector
            dummy_latlng = LatLng(math.degrees(dummy_coordinate_vector[0]), math.degrees(dummy_coordinate_vector[1]))
            dummy_node = Node(None, dummy_latlng)
            adj_street_nodes.insert(idx, dummy_node)

            # Create crosswalk nodes
            for i in range(len(adj_street_nodes)):
                n1 = adj_street_nodes[i - 1]
                n2 = adj_street_nodes[i]
                new_node = make_crosswalk_node(intersection_node, n1, n2)

                # Keep track of from which streets the crosswalk nodes are created.
                way_ids = []
                for wid in n1.get_way_ids():
                    way_ids.append(wid)
                for wid in n2.get_way_ids():
                    way_ids.append(wid)
                way_ids = list(set(intersection_node.get_way_ids()) & set(way_ids))  #list(set(way_ids))

                new_node.way_ids = way_ids
                sidewalk_nodes.add(new_node.id, new_node)
                new_crosswalk_node_ids.append(new_node.id)
                source_table[new_node.id] = [intersection_node, n1, n2]

            # Add a cross walk to the data structure
            new_crosswalk_node_ids.append(new_crosswalk_node_ids[0])
            crosswalk = Sidewalk(None, new_crosswalk_node_ids, "crosswalk")
            sidewalks.add(crosswalk.id, crosswalk)

            # Connect the new intersection sidewalk nodes with correct sidewalk nodes
            for crosswalk_node_ids in source_table:
                # Get the intersection node and two nodes that created the intersection sidewalk node
                ni, n1, n2 = source_table[crosswalk_node_ids]
                crosswalk_node = sidewalk_nodes.get(crosswalk_node_ids)
                v_crosswalk_node = ni.vector_to(crosswalk_node)  # A vector to the intersection sidewalk node

                for i, n_adj in enumerate([n1, n2]):
                    v_n_adj = ni.vector_to(n_adj)  # A vector from an intersection node to an adjacent street node

                    if len(set(crosswalk_node.get_way_ids())) == 1:
                        # The intersection sidewalk node was created from a dummy node and one
                        # adjacenct street node, thus there is only one way_id associated with
                        # the intersection sidewalk node (dummy node does not has a way).
                        shared_street_id = crosswalk_node.get_way_ids()[0]

                        if shared_street_id not in n_adj.sidewalk_nodes:
                            n_adj = n2
                    else:
                        shared_street_ids = set(intersection_node.get_way_ids()) & set(n_adj.get_way_ids())
                        shared_street_id = list(shared_street_ids)[0]

                    sidewalk_node_1_from_adj, sidewalk_node_2_from_adj = n_adj.sidewalk_nodes[shared_street_id]
                    v_n_adj_s1 = intersection_node.vector_to(sidewalk_node_1_from_adj)

                    # Check which one of n1_s1 and n1_s2 are on the same side of the road with crosswalk_node
                    # If the rotation (cross product) from v_n1 to v_crosswalk_node is same as v_n1 to v_n1_s1, then
                    # n1_s1 should be on the same side. Otherwise, n1_s1 should be on the same side with crosswalk_node.
                    if np.cross(v_n_adj, v_crosswalk_node) * np.cross(v_n_adj, v_n_adj_s1) > 0:  # (***)
                        sidewalk_node_from_adj = sidewalk_node_1_from_adj
                    else:
                        sidewalk_node_from_adj = sidewalk_node_2_from_adj

                    # Identify on which sidewalk n_adj_sidewalk_temp belongs too.
                    sidewalk_id = sidewalk_node_from_adj.get_way_ids()[0]
                    sidewalk = sidewalks.get(sidewalk_id)

                    # Swap the sidewalk intersection_sidewalk_node with crosswalk_node

                    potential_nodes_to_swap = [n.id for n in ni.get_sidewalk_nodes(shared_street_id)]
                    intersection_sidewalk_node_ids = set(sidewalk.nids) & set(potential_nodes_to_swap)

                    # Todo: I cannot figure out why I need this code snippets... (Issue #6)
                    # But the condition (***) sometimes identifies the wrong n_adj_sidewalk_temp
                    # This is probably due to the fact that distance between a sidewalk and
                    if len(intersection_sidewalk_node_ids) == 0:
                        if sidewalk_node_from_adj == sidewalk_node_1_from_adj:
                            sidewalk_node_from_adj = sidewalk_node_2_from_adj
                        else:
                            sidewalk_node_from_adj = sidewalk_node_1_from_adj
                        sidewalk_id = sidewalk_node_from_adj.get_way_ids()[0]
                        sidewalk = sidewalks.get(sidewalk_id)
                        potential_nodes_to_swap = [n.id for n in ni.get_sidewalk_nodes(shared_street_id)]
                        intersection_sidewalk_node_ids = set(sidewalk.nids) & set(potential_nodes_to_swap)

                    intersection_sidewalk_node_id = list(intersection_sidewalk_node_ids)[0]
                    # intersection_sidewalk_node_id_index = sidewalk.nids.index(intersection_sidewalk_node_id)
                    # sidewalk.nids[intersection_sidewalk_node_id_index] = crosswalk_node.id
                    # sidewalk_nodes.remove(intersection_sidewalk_node_id)

                    swap_nodes(sidewalk_nodes, sidewalk, intersection_sidewalk_node_id, crosswalk_node.id)

                    if len(set(crosswalk_node.get_way_ids())) == 1:
                        break

        else:
            # There are more than 3 nodes connected to the intersection node
            for i in range(len(adj_street_nodes)):
                n1 = adj_street_nodes[i - 1]
                n2 = adj_street_nodes[i]
                new_node = make_crosswalk_node(intersection_node, n1, n2)
                sidewalk_nodes.add(new_node.id, new_node)
                new_crosswalk_node_ids.append(new_node.id)
                source_table[new_node.id] = [intersection_node, n1, n2]

            # Add a cross walk to the data structure
            new_crosswalk_node_ids.append(new_crosswalk_node_ids[0])
            crosswalk = Sidewalk(None, new_crosswalk_node_ids, "crosswalk")
            sidewalks.add(crosswalk.id, crosswalk)

            # Connect the new intersection sidewalk nodes with correct sidewalk nodes
            for crosswalk_node_ids in source_table:
                # Get the intersection node and two nodes that created the crosswalk node
                ni, n1, n2 = source_table[crosswalk_node_ids]
                crosswalk_node = sidewalk_nodes.get(crosswalk_node_ids)
                v_crosswalk_node = ni.vector_to(crosswalk_node)  # A vector to the intersection sidewalk node

                # Connect sidewalk nodes created from each street node n1 and n2
                for n_adj in [n1, n2]:
                    # Get sidewalk nodes that are created from the street node, and
                    # identify which one should be connected to crosswalk_node
                    v_n_adj = ni.vector_to(n_adj)  # A vector from an intersection node to an adjacent street node
                    shared_street_ids = set(intersection_node.way_ids) & set(n_adj.way_ids)
                    shared_street_id = list(shared_street_ids)[0]  # Issue #7

                    # Get a pair of vectors to two sidewalk nodes created from n1
                    n_adj_s1, n_adj_s2 = n_adj.sidewalk_nodes[shared_street_id]
                    v_n_adj_s1 = intersection_node.vector_to(n_adj_s1)

                    # Check which one of n1_s1 and n1_s2 are on the same side of the road with crosswalk_node
                    # If the rotation (cross product) from v_n1 to v_crosswalk_node is same as v_n1 to v_n1_s1, then
                    # n1_s1 should be on the same side. Otherwise, n1_s1 should be on the same side with crosswalk_node.
                    if np.cross(v_n_adj, v_crosswalk_node) * np.cross(v_n_adj, v_n_adj_s1) > 0:
                        n_adj_sidewalk_temp = n_adj_s1
                    else:
                        n_adj_sidewalk_temp = n_adj_s2

                    # Identify on which sidewalk n_adj_sidewalk_temp belongs too.
                    sidewalk_id = n_adj_sidewalk_temp.get_way_ids()[0]
                    sidewalk = sidewalks.get(sidewalk_id)

                    # Swap the sidewalk intersection_sidewalk_node with crosswalk_node
                    intersection_sidewalk_node_ids = set(sidewalk.nids) & set([n.id for n in ni.get_sidewalk_nodes(shared_street_id)])

                    # Todo: I cannot figure out why I need this code snippets... (Issue #6)
                    # But the condition (***) sometimes identifies the wrong n_adj_sidewalk_temp
                    # This is probably due to the fact that distance between a sidewalk and
                    if len(intersection_sidewalk_node_ids) == 0:
                        if sidewalk_node_from_adj == sidewalk_node_1_from_adj:
                            sidewalk_node_from_adj = sidewalk_node_2_from_adj
                        else:
                            sidewalk_node_from_adj = sidewalk_node_1_from_adj
                        sidewalk_id = sidewalk_node_from_adj.get_way_ids()[0]
                        sidewalk = sidewalks.get(sidewalk_id)
                        potential_nodes_to_swap = [n.id for n in ni.get_sidewalk_nodes(shared_street_id)]
                        intersection_sidewalk_node_ids = set(sidewalk.nids) & set(potential_nodes_to_swap)

                    intersection_sidewalk_node_id = list(intersection_sidewalk_node_ids)[0]
                    # intersection_sidewalk_node_id_index = sidewalk.nids.index(intersection_sidewalk_node_id)
                    # sidewalk.nids[intersection_sidewalk_node_id_index] = crosswalk_node.id
                    # sidewalk_nodes.remove(intersection_sidewalk_node_id)
                    swap_nodes(sidewalk_nodes, sidewalk, intersection_sidewalk_node_id, crosswalk_node.id)

    return sidewalk_nodes, sidewalks


def main(street_nodes, streets):
    sidewalk_nodes, sidewalks = make_sidewalks(street_nodes, streets)
    osm = OSM(sidewalk_nodes, sidewalks)
    make_crosswalks(street_nodes, osm.nodes, streets, osm.ways)

    output = osm.export(format='geojson')
    print output


if __name__ == "__main__":
    #filename = "../resources/Simple4WayIntersection_01.osm"
    #filename = "../resources/Simple4WayIntersection_02.osm"
    # filename = "../resources/TShapeIntersection_01.osm"
    # filename = "../resources/TShapeIntersection_02.osm"
    #filename = "../resources/SegmentedStreet_01.osm"
    #filename = "../resources/ComplexIntersection_01.osm"
    #filename = "../resources/SmallMap_01.osm"
    filename = "../resources/SmallMap_03.osm"
    # filename = "../resources/ParallelLanes_01.osm"
    nodes, ways = parse(filename)
    osm_obj = OSM(nodes, ways)
    osm_obj.parse_intersections()

    # output = osm_obj.export()
    # print output
    main(osm_obj.nodes, osm_obj.ways)
