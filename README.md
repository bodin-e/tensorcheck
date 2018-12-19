# tensorcheck

tensorcheck is a (very small) library for validating tensors when using TensorFlow. The library is intended as a
demonstration of a suggested approach rather than a library in its own right, although contributions or improvement suggestions to the library are welcome.
The same approach for PyTorch or other frameworks having the tensor as the primary abstraction can be achieved with
 trivial adaptation. Authors: Erik Bodin, Andrew Lawrence

## Example
```python

import tensorflow as tf
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
    x=tf.constant(np.ones([10, 2])),
    y=tf.constant(np.ones([10, 1])),
    param=tf.constant(np.ones([2])),
    other_param=tf.constant(np.ones([2]))
)

# asserts true
m = model(
    x=tf.placeholder(shape=[None, 2], dtype=tf.float32),
    y=tf.placeholder(shape=[None, 1], dtype=tf.float32),
    param=tf.constant(np.ones([2])),
    other_param=tf.constant(np.ones([2]))
)

# asserts false
m = model(
    x=tf.constant(np.ones([10, 2])),
    y=tf.constant(np.ones([1, 10])),
    param=tf.constant(np.ones([2])),
    other_param=tf.constant(np.ones([2]))
)
# => AssertionError: Tensor 'y' dim 0 was of size 1 but was expected to be 10 as declared by 'x' dim 0

# asserts false
m = model(
    x=tf.constant(np.ones([10, 2])),
    y=tf.constant(np.ones([10, 1])),
    param=tf.constant(np.ones([2])),
    other_param=tf.constant(np.ones([3]))
)
# => AssertionError: Tensor 'other param' dim 0 was of size 3 but was expected to be 2 as declared directly

# asserts false
m = model(
    x=tf.constant(np.ones([10, 2])),
    y=tf.constant(np.ones([10, 1])),
    param=tf.constant(np.ones([2, 1])),
    other_param=tf.constant(np.ones([2]))
)
# => AssertionError: Tensor 'param' was declared to have 1 tensor dim(s) but had 2.

# asserts false
m = model(
    x=tf.constant(np.ones([10, 2])),
    y=tf.constant(np.ones([10, 1])),
    param=tf.constant(np.ones([1])),
    other_param=tf.constant(np.ones([2]))
)
# => AssertionError: Tensor 'param' dim 0 was of size 1 but was expected to be 2 as declared by 'x' dim 1
...
```
