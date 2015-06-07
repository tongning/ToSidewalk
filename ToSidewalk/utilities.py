from itertools import islice
import numpy as np

def area(p1, p2, p3):
    """
    Given 3 points (x1, y1), (x2, y2), (x3, y3), return the area.
    :param p1:
    :param p2:
    :param p3:
    :return:
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    area = np.cross(v1, v2) / 2
    return abs(area)

def window(seq, n=2, padding=None):
    """
    Itertools
    https://docs.python.org/2/library/itertools.html#recipes
    Helper sliding window iterater method
    See: http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python

    Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
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
    return
