from pygrb.utils import get_first_intersection, is_iterable, extract_number
from numpy import array

def test_get_first_intersection():
    x = array([1, 2, 3, 4, 5])
    y = array([2, 4, 6, 8, 10])
    value = 5
    assert get_first_intersection(x, y, value) == 3

def test_is_iterable():
    assert is_iterable([1, 2, 3]) == True
    assert is_iterable((1, 2, 3)) == True
    assert is_iterable({1, 2, 3}) == True
    assert is_iterable('123') == True
    assert is_iterable(123) == False
    assert is_iterable(None) == False

def test_extract_number():
    assert extract_number('123') == 123
    assert extract_number('3.14') == 3.14
    assert extract_number('hello') == 'hello'