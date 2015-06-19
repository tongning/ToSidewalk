from itertools import islice
import numpy as np

def area(p1, p2, p3):
    """
    Given three points (x1, y1), (x2, y2), (x3, y3), return the area of the triangle that is formed by the three points.
    :param p1: Point 1 (e.g., [x1, y1])
    :param p2: Point 2 (e.g., [x2, y2])
    :param p3: Point 3 (e.g., [x3, y3])
    :return: Area of a triangle
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    area = np.cross(v1, v2) / 2
    return abs(area)

def window(seq, n=2, padding=None):
    """
    Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...

    Itertools
    https://docs.python.org/2/library/itertools.html#recipes
    Helper sliding window iterater method
    See: http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python

    :param seq: An iterable like a list
    :param n: A size of a window
    :param padding: Padding at the both ends of the iterable.
    """
    # it = iter(seq)
    if padding is not None:
        seq = [None for i in range(padding)] + list(seq) + [None for i in range(padding)]
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
