
import tensorflow as tf
import numpy as np
from tensorcheck.static import assert_shapes


def model(x, y, param):
    assert_shapes({
        ("x", x): ('N', 'Q'),
        ("y", y): ('N', 'D'),
        ("param", param): 'Q',
    })
    ...

# asserts true
m = model(
    x=tf.constant(np.ones([10, 2])),
    y=tf.constant(np.ones([10, 1])),
    param=tf.constant(np.ones([2]))
)

# asserts true
m = model(
    x=tf.placeholder(shape=[None, 2], dtype=tf.float32),
    y=tf.placeholder(shape=[None, 1], dtype=tf.float32),
    param=tf.constant(np.ones([2]))
)

# asserts false
m = model(
    x=tf.constant(np.ones([10, 1])),
    y=tf.constant(np.ones([2, 10])),
    param=tf.constant(np.ones([2]))
)
# => AssertionError: Tensor 'y' dim 0 was of size 2 but was expected to be 10 as declared by 'x' dim 0

# asserts false
m = model(
    x=tf.constant(np.ones([10, 2])),
    y=tf.constant(np.ones([10, 1])),
    param=tf.constant(np.ones([2, 1]))
)
# => AssertionError: Tensor 'param' was declared to have 1 tensor dim(s) but have 2.

# asserts false
m = model(
    x=tf.constant(np.ones([10, 2])),
    y=tf.constant(np.ones([10, 1])),
    param=tf.constant(np.ones([1]))
)
# => AssertionError: Tensor 'param' dim 0 was of size 1 but was expected to be 2 as declared by 'x' dim 1