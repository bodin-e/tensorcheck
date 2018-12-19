
import tensorflow as tf
import numpy as np
import tensorcheck.static as static
import hypothesis.strategies as st
from hypothesis import given
import pytest
import random


@given(
    st.lists(
        st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=3),
        min_size=1,
        max_size=5
    )
)
def test_static_values_correct(sizes):
    declarations = {
        ("tensor %s" % i, tf.constant(np.ones(tensor_size))): tensor_size
        for i, tensor_size in enumerate(sizes)
    }
    static.assert_shapes(declarations)


@given(
    st.lists(
        st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=3),
        min_size=1,
        max_size=5
    ),
    st.integers(min_value=0, max_value=3)
)
def test_static_values_incorrect(sizes, mutate_index):
    def mutate(tensor_size):
        mutated_size = tensor_size.copy()
        mutated_size[mutate_index % len(tensor_size)] += 1
        return mutated_size
    declarations = {
        ("tensor %s" % i, tf.constant(np.ones(tensor_size))): mutate(tensor_size)
        for i, tensor_size in enumerate(sizes)
    }
    with pytest.raises(AssertionError):
        static.assert_shapes(declarations)


def tensor_generator(symbol_value_map):
    return lambda symbols: tf.constant(np.ones([symbol_value_map[s] for s in symbols]))


def test_declared_symbols_varying_num_tensors_and_dims():
    max_num_dims = 10
    max_num_tensors = 10

    symbol_value_map = {
        chr(i): i
        for i in range(1, max_num_dims)
    }

    tensor = tensor_generator(symbol_value_map)

    num_tensors_configs = sorted(range(1, max_num_tensors), key=lambda k: random.random())
    num_dims_configs = sorted(range(1, max_num_dims), key=lambda k: random.random())

    for num_tensors in num_tensors_configs:
        for num_dims in num_dims_configs:
            symbols = [chr(i) for i in range(1, num_dims)]
            static.assert_shapes({
                (chr(t), tensor(symbols)): symbols
                for t in range(1, num_tensors)
            })


def test_declared_values_correct():
    symbol_value_map = {
        'A': 1,
        'B': 2,
        'C': 3,
        'D': 4,
        'E': 5,
    }
    tensor = tensor_generator(symbol_value_map)
    static.assert_shapes({
        ("a", tensor(['A'])): 'A',
        ("b", tensor(['B'])): 'B',
        ("c", tensor(['A', 'B'])): ('A', 'B'),
    })
    static.assert_shapes({
        ("a", tensor(['A'])): 'A',
        ("b", tensor(['C', 'A'])): ('C', 'A'),
    })
    static.assert_shapes({
        ("a", tensor(['A', 'B'])): ('A', 'B'),
        ("b", tensor(['C', 'D'])): ('C', 'D'),
    })
    static.assert_shapes({
        ("a", tensor(['A', 'B'])): ('A', 'B'),
        ("b", tensor(['A', 'C'])): ('A', 'C'),
        ("c", tensor(['B', 'D'])): ('B', 'D'),
        ("d", tensor(['E', 'C'])): ('E', 'C'),
    })
    static.assert_shapes({
        ("a", tensor(['A', 'B'])): ('A', 'B'),
        ("b", tensor(['A', 'B', 'C'])): ('A', 'B', 'C'),
        ("c", tensor(['C', 'B', 'A'])): ('C', 'B', 'A'),
        ("d", tensor(['C'])): 'C',
    })


def test_declared_values_incorrect():
    symbol_value_map = {
        'A': 1,
        'B': 2,
        'C': 3,
        'D': 4,
        'E': 5,
    }
    tensor = tensor_generator(symbol_value_map)
    with pytest.raises(AssertionError):
        static.assert_shapes({
            ("a", tensor(['A'])): 'A',
            ("b", tensor(['B'])): 'B',
            ("b", tensor(['A', 'B'])): ('B', 'A'),
        })
    with pytest.raises(AssertionError):
        static.assert_shapes({
            ("a", tensor(['A'])): 'A',
            ("a", tensor(['B'])): 'A',
        })
    with pytest.raises(AssertionError):
        static.assert_shapes({
            ("a", tensor(['A'])): ('A', 1),
        })
    with pytest.raises(AssertionError):
        static.assert_shapes({
            ("a", tensor(['A'])): 'A',
            ("b", tensor(['B'])): 'B',
            ("c", tensor(['A', 'B'])): ('A', 'B', 1),
        })
    with pytest.raises(AssertionError):
        static.assert_shapes({
            ("a", tensor(['A'])): 'A',
            ("b", tensor(['A', 'B'])): ('A', 'B'),
            ("c", tensor(['A', 'B', 'C'])): ('A', 'C', 'B'),
        })
    with pytest.raises(AssertionError):
        static.assert_shapes({
            ("a", tensor(['A'])): 'A',
            ("b", tensor(['A', 'B'])): ('A', 'B'),
            ("c", tensor(['A', 'B', 'C'])): ('A', 'B', 'C'),
            ("d", tensor(['D', 'A', 'B'])): ('D', 'B', 'B'),
        })