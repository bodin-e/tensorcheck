
import tensorflow as tf
import torch
import numpy as np
from tensorcheck.static import assert_shapes


def model(x, y, param, other_param):
    assert_shapes({
        ("x", x): ('N', 'Q'),
        ("y", y): ('N', 'D'),
        ("param", param): 'Q',
        ("other param", other_param): 2,
    })
    ...

# asserts true
m = model(
    x=tf.ones([10, 2]),
    y=tf.ones([10, 1]),
    param=tf.ones([2]),
    other_param=tf.ones([2])
)

# asserts true
m = model(
    x=torch.ones([10, 2]),
    y=torch.ones([10, 1]),
    param=torch.ones([2]),
    other_param=torch.ones([2])
)

# asserts true
m = model(
    x=tf.placeholder(shape=[None, 3], dtype=tf.float32),
    y=tf.placeholder(shape=[None, 1], dtype=tf.float32),
    param=tf.ones([3]),
    other_param=tf.ones([2])
)

# asserts false
m = model(
    x=tf.constant(np.ones([10, 2])),
    y=tf.constant(np.ones([1, 10])),
    param=tf.ones([2]),
    other_param=tf.ones([2])
)
# => AssertionError: Tensor 'y' dim 0 was of size 1 but was expected to be 10 as declared by 'x' dim 0

# asserts false
m = model(
    x=tf.constant(np.ones([10, 2])),
    y=tf.constant(np.ones([10, 1])),
    param=tf.ones([2]),
    other_param=tf.ones([3])
)
# => AssertionError: Tensor 'other param' dim 0 was of size 3 but was expected to be 2 as declared directly

# asserts false
m = model(
    x=tf.ones([10, 2]),
    y=tf.ones([10, 1]),
    param=tf.ones([2, 1]),
    other_param=tf.ones([2])
)
# => AssertionError: Tensor 'param' was declared to have 1 tensor dim(s) but had 2.

# asserts false
m = model(
    x=tf.ones([10, 2]),
    y=tf.ones([10, 1]),
    param=tf.ones([1]),
    other_param=tf.ones([2])
)
# => AssertionError: Tensor 'param' dim 0 was of size 1 but was expected to be 2 as declared by 'x' dim 1