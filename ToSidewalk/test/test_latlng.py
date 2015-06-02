import unittest
import math
from ToSidewalk.latlng import haversine

class TestLatLngMethods(unittest.TestCase):
    def test_haversine(self):
        """
        Haversine ground truth is from:
        http://andrew.hedges.name/experiments/haversine/
        """
        latlng1 = [38.898556, -77.037852]
        latlng2 = [38.897147, -77.043934]
        latlng1 = [math.radians(x) for x in latlng1]
        latlng2 = [math.radians(x) for x in latlng2]

        distance = 0.549
        error = abs(distance - haversine(latlng1[1], latlng1[0], latlng2[1], latlng2[0]))
        self.assertTrue(error < 0.001)

        latlng1 = [38.900665, -76.983008]
        latlng2 = [38.9007234, -76.98197]
        latlng1 = [math.radians(x) for x in latlng1]
        latlng2 = [math.radians(x) for x in latlng2]
        distance = 0.09

        error = abs(distance - haversine(latlng1[1], latlng1[0], latlng2[1], latlng2[0]))
        self.assertTrue(error < 0.001)

if __name__ == '__main__':
    unittest.main()