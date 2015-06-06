from xml.etree import cElementTree as ET
from shapely.geometry import Polygon, Point, LineString
import numpy as np
import math

from latlng import LatLng
from nodes import Node, Nodes
from ways import Street, Streets


class Network(object):
    def __init__(self, nodes, ways):
        self.nodes = nodes
        self.ways = ways

        self.bounds = [100000.0, 100000.0, -1.0, -1.0]  # min lat, min lng, max lat, and max lng

        # Initialize the bounding box
        for node in self.nodes.get_list():
            lat, lng = node.latlng.location(radian=False)
            if lat < self.bounds[0]:
                self.bounds[0] = lat
            elif lat > self.bounds[2]:
                self.bounds[2] = lat
            if lng < self.bounds[1]:
                self.bounds[1] = lng
            elif lng > self.bounds[3]:
                self.bounds[3] = lng

    def get_adjacent_nodes(self, node):
        """
        Get adjacent nodes for the passed node
        :param node:
        :return:
        """
        adj_nodes = []
        way_ids = node.get_way_ids()

        for way_id in way_ids:
            way = self.ways.get(way_id)
            # If the current intersection node is at the head of street.nids, then take the second node and push it
            # into adj_street_nodes. Otherwise, take the node that is second to the last in street.nids .
            if way.nids[0] == node.id:
                adj_nodes.append(self.nodes.get(way.nids[1]))
            else:
                adj_nodes.append(self.nodes.get(way.nids[-2]))

        return adj_nodes

    def parse_intersections(self):
        parse_intersections(self.nodes, self.ways)
        return

class OSM(Network):

    def __init__(self, nodes, ways):
        # self.nodes = nodes
        # self.ways = ways
        super(OSM, self).__init__(nodes, ways)

        # self.bounds = [100000.0, 100000.0, -1.0, -1.0]  # min lat, min lng, max lat, and max lng

    def preprocess(self):
        # Preprocess and clean up the data

        self.merge_parallel_street_segments()

        self.split_streets()
        self.update_ways()
        self.merge_nodes()
        # Clean up and so I can make a sidewalk network
        self.clean_up_nodes()
        self.clean_street_segmentation()

        # Remove ways that have only a single node.
        for way in self.ways.get_list():
            if len(way.nids) < 2:
                for nid in way.get_node_ids():
                    n = self.nodes.get(nid)
                    n.remove_way_id(way.id)
                self.ways.remove(way.id)
        return

    def clean_up_nodes(self):
        """
        Remove unnecessary nodess
        """
        nids = []
        for way in self.ways.get_list():
            nids.extend(way.nids)
        nids = set(nids)

        new_nodes = Nodes()
        for nid in nids:
            new_nodes.add(nid, self.nodes.get(nid))

        self.nodes = new_nodes
        return

    def clean_street_segmentation(self):
        """
        Go through nodes and find ones that have two connected ways (nodes should have either one or more than two ways)
        """
        for node in self.nodes.get_list():
            if len(node.get_way_ids()) == 2:
                way_id_1, way_id_2 = node.get_way_ids()
                way_1 = self.ways.get(way_id_1)
                way_2 = self.ways.get(way_id_2)

                # Given that the streets are split, node's index in each way's nids (a list of node ids) should
                # either be 0 or else.
                combined_nids = []
                if way_1.nids.index(node.id) == 0 and way_2.nids.index(node.id) == 0:
                    combined_nids = way_1.nids[:0:-1] + way_2.nids
                if way_1.nids.index(node.id) != 0 and way_2.nids.index(node.id) == 0:
                    combined_nids = way_1.nids[:-1] + way_2.nids
                if way_1.nids.index(node.id) == 0 and way_2.nids.index(node.id) != 0:
                    combined_nids = way_2.nids[:-1] + way_1.nids
                else:
                    combined_nids = way_1.nids + way_2.nids[1::-1]

                # Create a new way from way_1 and way_2. Then remove the two ways from self.way
                new_street = Street(None, combined_nids, "footway")
                self.ways.add(new_street.id, new_street)
                self.ways.remove(way_id_1)
                self.ways.remove(way_id_2)

        return

    def export(self, format="geojson"):
        """
        Export the node and way data.
        Todo: Implement geojson format for export.
        """
        if format == 'osm':
            header = """
<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
<bounds minlat="%s" minlon="%s" maxlat="%s" maxlon="%s" />
""" % (str(self.bounds[0]), str(self.bounds[1]), str(self.bounds[2]), str(self.bounds[3]))

            footer = "</osm>"
            node_list = []
            for node in self.nodes.get_list():
                lat, lng = node.latlng.location(radian=False)
                node_str = """<node id="%s" visible="true" user="test" lat="%s" lon="%s" />""" % (str(node.id), str(lat), str(lng))
                node_list.append(node_str)

            way_list = []
            for way in self.ways.get_list():
                way_str = """<way id="%s" visible="true" user="test">""" % (str(way.id))
                way_list.append(way_str)
                for nid in way.get_node_ids():
                    nid_str = """<nd ref="%s" />""" % (str(nid))
                    way_list.append(nid_str)

                if way.type is not None:
                    tag = """<tag k="%s" v="%s" />""" % ("highway", way.type)
                    if way.type == "footway":
                        # How to tag sidewalks in OpenStreetMap
                        # https://help.openstreetmap.org/questions/1236/should-i-map-sidewalks
                        # http://wiki.openstreetmap.org/wiki/Tag:footway%3Dsidewalk
                        tag = """<tag k="%s" v="%s" />""" % ("footway", "sidewalk")
                    way_list.append(tag)
                way_list.append("</way>")

            osm = header + "\n".join(node_list) + "\n" + "\n".join(way_list) + "\n" + footer

            return osm
        else:
            # Mapbox GeoJson format
            # https://github.com/mapbox/simplestyle-spec/tree/master/1.1.0
            geojson = {}
            geojson['type'] = "FeatureCollection"
            geojson['features'] = []
            for way in self.ways.get_list():
                feature = {}
                feature['properties'] = {
                    'type': way.type,
                    'id': way.id,
                    'user': way.user,
                    'stroke': '#555555'
                }
                feature['type'] = 'Feature'
                feature['id'] = 'way/%s' % (way.id)


                coordinates = []
                for nid in way.nids:
                    latlng = self.nodes.get(nid).latlng
                    coordinates.append([latlng.lng, latlng.lat])
                feature['geometry'] = {
                    'type': 'LineString',
                    'coordinates': coordinates
                }
                geojson['features'].append(feature)

            import json
            return json.dumps(geojson)

    def merge_nodes(self, distance_threshold=0.015):
        """
        Merge nodes that are close to intersection nodes. Then merge nodes that are
        close to each other.
        """
        for street in self.ways.get_list():
            # if len(street.nids) < 2:
            if len(street.nids) <= 2:
                # Skip. You should not merge two intersection nodes
                continue

            start = self.nodes.get(street.nids[0])
            end = self.nodes.get(street.nids[-1])

            # Merge the nodes around the beginning of the street
            for nid in street.nids[1:-1]:
                target = self.nodes.get(nid)
                distance = start.distance_to(target)
                if distance < distance_threshold:
                    street.nids.remove(nid)
                else:
                    break

            if len(street.nids) <= 2:
                # Done, if you merged everything other than intersection nodes
                continue

            for nid in street.nids[-2:0:-1]:
                target = self.nodes.get(nid)
                distance = end.distance_to(target)
                if distance < distance_threshold:
                    street.nids.remove(nid)
                else:
                    break

        return

    def merge_parallel_street_segments(self):
        """
        Todo: This method needs to be optimized using some spatial data structure (e.g., r*-tree) and other metadata..
        Todo. And I should break this function into find_parallel_street_segments and merge_parallel_street_segments.
        # Expand streets into rectangles, then find intersections between them.
        # http://gis.stackexchange.com/questions/90055/how-to-find-if-two-polygons-intersect-in-python
        """

        streets = self.ways.get_list()
        street_polygons = []
        distance_to_sidewalk = 0.0003

        for street in streets:
            start_node_id = street.get_node_ids()[0]
            end_node_id = street.get_node_ids()[-1]
            start_node = self.nodes.get(start_node_id)
            end_node = self.nodes.get(end_node_id)

            vector = start_node.vector_to(end_node, normalize=True)
            perpendicular = np.array([vector[1], - vector[0]]) * distance_to_sidewalk
            p1 = start_node.vector() + perpendicular
            p2 = end_node.vector() + perpendicular
            p3 = end_node.vector() - perpendicular
            p4 = start_node.vector() - perpendicular

            poly = Polygon([p1, p2, p3, p4])
            poly.angle = math.degrees(math.atan2(vector[0], vector[1]))
            poly.nids = set((start_node_id, end_node_id))
            street_polygons.append(poly)

        # Find pair of polygons that intersect each other.
        from itertools import combinations
        polygon_combinations = combinations(street_polygons, 2)
        parallel_pairs = []
        for pair in polygon_combinations:
            angle_diff = ((pair[0].angle - pair[1].angle) + 180.) % 180.
            if pair[0].intersects(pair[1]) and angle_diff < 10.:
                # If the polygon intersects, and they have a kind of similar angle, and they don't share a node,
                # then they should be merged together.

                parallel_pairs.append((street_polygons.index(pair[0]), street_polygons.index(pair[1])))

        # Merge parallel pairs
        for pair in parallel_pairs:
            street_pair = (streets[pair[0]], streets[pair[1]])

            shared_nids = set(street_pair[0].nids) & set(street_pair[1].nids)

            # Find the adjacent nodes for the shared node
            if len(shared_nids) > 0:
                # Two paths merges at one node
                shared_nid = list(shared_nids)[0]
                shared_node = self.nodes.get(shared_nid)
                idx1 = street_pair[0].nids.index(shared_nid)
                idx2 = street_pair[1].nids.index(shared_nid)

                # Nodes are sorted by longitude (x-axis), so two paths should merge at the left-most node or the
                # right most node.
                if idx1 == 0 and idx2 == 0:
                    adj_nid1 = street_pair[0].nids[1]
                    adj_nid2 = street_pair[1].nids[1]
                else:
                    adj_nid1 = street_pair[0].nids[-2]
                    adj_nid2 = street_pair[1].nids[-2]

                adj_node1 = self.nodes.get(adj_nid1)
                adj_node2 = self.nodes.get(adj_nid2)
                angle_to_node1 = math.degrees(shared_node.angle_to(adj_node1))
                angle_to_node2 = math.degrees(shared_node.angle_to(adj_node2))
                if abs(angle_to_node2 - angle_to_node1) > 90:
                    # Paths are connected but they are not parallel lines
                    continue

            # First find parts of the streets that you want to merge (you don't want to merge entire streets
            # because, for example, one could be much longer than the other.

            # Take the first two points of street_pair[0], and use it as a base vector.
            # Project all the points along the base vector and sort them.
            base_node0 = self.nodes.get(street_pair[0].nids[0])
            base_node1 = self.nodes.get(street_pair[0].nids[-1])
            base_vector = base_node0.vector_to(base_node1, normalize=True)
            def cmp_with_projection(n1, n2):
                dot_product1 = np.dot(n1.vector(), base_vector)
                dot_product2 = np.dot(n2.vector(), base_vector)
                if dot_product1 < dot_product2:
                    return -1
                elif dot_product2 < dot_product1:
                    return 1
                else:
                    return 0
            all_nodes = [self.nodes.get(nid) for nid in street_pair[0].nids] + [self.nodes.get(nid) for nid in street_pair[1].nids]
            all_nodes = sorted(all_nodes, cmp=cmp_with_projection)
            all_nids = [node.id for node in all_nodes]

            # for node in all_nodes:
            #     print node.latlng
            # return

            # print all_nids

            # Condition in list comprehension
            # http://stackoverflow.com/questions/4260280/python-if-else-in-list-comprehension
            all_nids_street_indices = [0 if nid in street_pair[0].nids else 1 for nid in all_nids]
            from utilities import window
            all_nids_street_switch = [idx_pair[0] != idx_pair[1] for idx_pair in window(all_nids_street_indices, 2)]

            # Find the first occurence of an element in a list
            # http://stackoverflow.com/questions/9868653/find-first-list-item-that-matches-criteria
            begin_idx = all_nids_street_switch.index(next(x for x in all_nids_street_switch if x == True))

            # Find the last occurence of an element in a list
            # http://stackoverflow.com/questions/6890170/how-to-find-the-last-occurrence-of-an-item-in-a-python-list
            end_idx = (len(all_nids_street_switch) - 1) - all_nids_street_switch[::-1].index(next(x for x in all_nids_street_switch if x == True))

            # print all_nids_street_indices[begin_idx:end_idx + 2]
            # print all_nids[begin_idx:end_idx + 2]

            # Find the parallel part of the two segments.
            subset_nids = all_nids[begin_idx:end_idx + 1]
            if subset_nids[0] in streets[pair[0]].nids:
                street1_idx = streets[pair[0]].nids.index(subset_nids[0])
                street2_idx = streets[pair[1]].nids.index(subset_nids[1])
            else:
                street1_idx = streets[pair[0]].nids.index(subset_nids[1])
                street2_idx = streets[pair[1]].nids.index(subset_nids[0])

            if subset_nids[-1] in streets[pair[0]].nids:
                street1_end_idx = streets[pair[0]].nids.index(subset_nids[-1])
                street2_end_idx = streets[pair[1]].nids.index(subset_nids[-2])
            else:
                street1_end_idx = streets[pair[0]].nids.index(subset_nids[-2])
                street2_end_idx = streets[pair[1]].nids.index(subset_nids[-1])

            # Get two parallel segments and the distance between them
            street1_nid = streets[pair[0]].nids[street1_idx]
            street2_nid = streets[pair[1]].nids[street2_idx]
            street1_end_nid = streets[pair[0]].nids[street1_end_idx]
            street2_end_nid = streets[pair[1]].nids[street2_end_idx]
            street1_node = self.nodes.get(street1_nid)
            street2_node = self.nodes.get(street2_nid)
            street1_end_node = self.nodes.get(street1_end_nid)
            street2_end_node = self.nodes.get(street2_end_nid)

            LS_street1 = LineString((street1_node.latlng.location(radian=False), street1_end_node.latlng.location(radian=False)))
            LS_street2 = LineString((street2_node.latlng.location(radian=False), street2_end_node.latlng.location(radian=False)))
            distance = LS_street1.distance(LS_street2) / 2

            # Merge streets
            for nid in subset_nids:
                if nid == street1_nid:
                    # print "Street 1"
                    street1_idx += 1
                    street1_nid = streets[pair[0]].nids[street1_idx]

                    node = self.nodes.get(nid)
                    opposite_node_1 = self.nodes.get(street2_nid)
                    opposite_node_2_nid = streets[pair[1]].nids[street2_idx + 1]
                    opposite_node_2 = self.nodes.get(opposite_node_2_nid)

                    v = opposite_node_1.vector_to(opposite_node_2, normalize=True)
                    v2 = opposite_node_1.vector_to(node, normalize=True)
                    if np.cross(v, v2) > 0:
                        normal = np.array([v[1], v[0]])
                    else:
                        normal = np.array([- v[1], v[0]])
                    new_position = node.latlng.location(radian=False) + normal * distance
                    node.latlng.lat, node.latlng.lng = new_position

                else:
                    # print "Street 2"
                    street2_idx += 1
                    street2_nid = streets[pair[1]].nids[street2_idx]

                    node = self.nodes.get(nid)
                    opposite_node_1 = self.nodes.get(street1_nid)
                    opposite_node_2_nid = streets[pair[0]].nids[street1_idx + 1]
                    opposite_node_2 = self.nodes.get(opposite_node_2_nid)

                    v = opposite_node_1.vector_to(opposite_node_2, normalize=True)
                    v2 = opposite_node_1.vector_to(node, normalize=True)
                    if np.cross(v, v2) > 0:
                        normal = np.array([v[1], v[0]])
                    else:
                        normal = np.array([- v[1], v[0]])
                    new_position = node.latlng.location(radian=False) + normal * distance
                    node.latlng.lat, node.latlng.lng = new_position

                # print node.latlng
        return


    def split_streets(self):
        """
        Split ways into pieces for preprocessing the OSM files.
        """
        new_streets = Streets()
        for way in self.ways.get_list():
            intersection_nids = [nid for nid in way.nids if self.nodes.get(nid).is_intersection()]
            intersection_indices = [way.nids.index(nid) for nid in intersection_nids]
            if len(intersection_indices) == 0:
                new_streets.add(way.id, way)
            else:
                # Do not split streets if (i) there is only one intersection node and it is the on the either end of the
                # street, or (ii) there are only two nodes and both of them are on the edge of the street.
                # Otherwise split the street!
                if len(intersection_indices) == 1 and (intersection_indices[0] == 0 or intersection_indices[0] == len(way.nids) - 1):
                    new_streets.add(way.id, way)
                elif len(intersection_indices) == 2 and (intersection_indices[0] == 0 and intersection_indices[1] == len(way.nids) - 1):
                    new_streets.add(way.id, way)
                elif len(intersection_indices) == 2 and (intersection_indices[1] == 0 and intersection_indices[0] == len(way.nids) - 1):
                    new_streets.add(way.id, way)
                else:
                    prev_idx = 0
                    for idx in intersection_indices:
                        if idx != 0 and idx != len(way.nids):
                            new_nids = way.nids[prev_idx:idx + 1]
                            new_way = Street(None, new_nids, way.type)
                            new_streets.add(new_way.id, new_way)
                            prev_idx = idx
                    new_nids = way.nids[prev_idx:]
                    new_way = Street(None, new_nids, way.type)
                    new_streets.add(new_way.id, new_way)
        self.ways = new_streets

        return

    def update_ways(self):
        # Update the way_ids
        for node in self.nodes.get_list():
            # Now the minimum number of ways connected has to be 3 for the node to be an intersection
            node.way_ids = []
            node.min_intersection_cardinality = 3
        for street in self.ways.get_list():
            for nid in street.nids:
                self.nodes.get(nid).append_way(street.id)


def parse(filename):
    """
    Parse a OSM file
    """
    with open(filename, "rb") as osm:
        # Find element
        # http://stackoverflow.com/questions/222375/elementtree-xpath-select-element-based-on-attribute
        tree = ET.parse(osm)

        nodes_tree = tree.findall(".//node")
        ways_tree = tree.findall(".//way")

    street_nodes = Nodes()
    for node in nodes_tree:
        mynode = Node(node.get("id"), LatLng(node.get("lat"), node.get("lon")))
        street_nodes.add(node.get("id"), mynode)

    # Parse ways and find streets that has the following tags
    streets = Streets()
    valid_highways = set(['primary', 'secondary', 'tertiary', 'residential'])
    for way in ways_tree:
        highway_tag = way.find(".//tag[@k='highway']")
        if highway_tag is not None and highway_tag.get("v") in valid_highways:
            node_elements = filter(lambda elem: elem.tag == "nd", list(way))
            nids = [node.get("ref") for node in node_elements]

            # Sort the nodes by longitude.
            if street_nodes.get(nids[0]).latlng.lng > street_nodes.get(nids[-1]).latlng.lng:
                nids = nids[::-1]

            street = Street(way.get("id"), nids)
            streets.add(way.get("id"), street)

    # Find intersections and store adjacency information
    for street in streets.get_list():
        # prev_nid = None
        for nid in street.nids:
            street_nodes.get(nid).append_way(street.id)

            # if street_nodes.get(nid).is_intersection() and nid not in streets.intersection_node_ids:
            #     streets.intersection_node_ids.append(nid)

    return street_nodes, streets


def parse_intersections(nodes, ways):
    node_list = nodes.get_list()
    intersection_node_ids = [node.id for node in node_list if node.is_intersection()]
    ways.set_intersection_node_ids(intersection_node_ids)
    return

if __name__ == "__main__":
    # filename = "../resources/SegmentedStreet_01.osm"
    filename = "../resources/ParallelLanes_01.osm"
    nodes, ways = parse(filename)
    street_network = OSM(nodes, ways)
    street_network.preprocess()
    street_network.parse_intersections()

    geojson = street_network.export(format='geojson')
    print geojson

