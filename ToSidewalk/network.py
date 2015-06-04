from xml.etree import cElementTree as ET

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

        # Preprocess and clean up the data
        self.split_streets()
        self.merge_parallel_street_segments()
        self.merge_nodes()
        self.clean_up_nodes()
        self.clean_street_segmentation()

        # Remove ways that have only a single node.
        for way in self.ways.get_list():
            if len(way.nids) < 2:
                for nid in way.get_node_ids():
                    n = self.nodes.get(nid)
                    n.remove_way_id(way.id)
                self.ways.remove(way.id)

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
        merge_parallel_street_segments(self.nodes, self.ways)
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

        # Update the way_ids
        for node in self.nodes.get_list():
            # Now the minimum number of ways connected has to be 3 for the node to be an intersection
            node.way_ids = []
            node.min_intersection_cardinality = 3
        for street in self.ways.get_list():
            for nid in street.nids:
                self.nodes.get(nid).append_way(street.id)
        return


def merge_parallel_street_segments(node, ways):

    return


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
    filename = "../resources/SegmentedStreet_01.osm"
    nodes, ways = parse(filename)
    osm_obj = OSM(nodes, ways)
    osm_obj.parse_intersections()

    geojson = osm_obj.export(format='geojson')

