class Way(object):
    def __init__(self, wid=None, nids=(), type=None):
        if wid is None:
            self.id = str(id(self))
        else:
            self.id = str(wid)
        self.nids = nids
        self.type = type
        self.user = 'test'
        self.parent_ways = None

    def belongs_to(self):
        return self.parent_ways

    def export(self):
        if self.parent_ways:
            pass

    def get_node_ids(self):
        return self.nids

    def get_shared_node_ids(self, other):
        """
        Other could be either a list of node ids or a Way object
        """
        if type(other) == list:
            return list(set(self.nids) & set(other))
        else:
            return list(set(self.nids) & set(other.get_node_ids()))


class Ways(object):
    def __init__(self):
        self.ways = {}
        self.intersection_node_ids = []
        self.parent_network = None

    def __eq__(self, other):
        return id(self) == id(other)

    def add(self, way):
        way.parent_ways = self
        self.ways[way.id] = way

    def belongs_to(self):
        return self.parent_network

    def get(self, wid):
        return self.ways[wid]

    def get_list(self):
        return self.ways.values()

    def remove(self, wid):
        # http://stackoverflow.com/questions/5844672/delete-an-element-from-a-dictionary
        del self.ways[wid]
        return

    def set_intersection_node_ids(self, nids):
        self.intersection_node_ids = nids

# Notes on inheritance
# http://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
class Street(Way):
    def __init__(self, wid=None, nids=(), type=None):
        super(Street, self).__init__(wid, nids, type)
        self.sidewalk_ids = []  # Keep track of which sidewalks were generated from this way
        self.distance_to_sidewalk = 0.000001

    def append_sidewalk_id(self, way_id):
        self.sidewalk_ids.append(way_id)
        return self

    def get_sidewalk_ids(self):
        return self.sidewalk_ids

class Streets(Ways):
    def __init__(self):
        super(Streets, self).__init__()

class Sidewalk(Way):
    def __init__(self, wid=None, nids=(), type=None):
        super(Sidewalk, self).__init__(wid, nids, type)

    def set_street_id(self, street_id):
        """  Set the parent street id """
        self.street_id = street_id
        return

    def swap_nodes(self, nid_from, nid_to):
        index_from = self.nids.index(nid_from)
        self.nids[index_from] = nid_to

class Sidewalks(Ways):
    def __init__(self):
        super(Sidewalks, self).__init__()
        self.street_id = None

