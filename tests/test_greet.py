import pytest

from inferai.core import greet


def test_greet():
    assert greet("Alice") == "Hello, Alice!"


def test_greet_empty():
    with pytest.raises(ValueError):
        greet("")
