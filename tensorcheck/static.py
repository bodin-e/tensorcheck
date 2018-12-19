
import collections

MESSAGE_NOT_SAME_NUM_TENSOR_DIMS_AS_DECLARED = \
    "Tensor '%s' was declared to have %s tensor dim(s) but had %s."
MESSAGE_NOT_SAME_DIM_SIZE_AS_DECLARED_BY = \
    "Tensor '%s' dim %s was of size %s but was expected to be %s as declared by '%s' dim %s"
MESSAGE_NOT_CORRECT_STATIC_DIM_SIZE = \
    "Tensor '%s' dim %s was of size %s but was expected to be %s as declared directly"


def assert_shapes(declarations):
    """
    Validate tensor shapes are matching as specified.
    Example:
    assert_shapes({
        ('x', x): ('N', 'Q'),
        ('y', y): ('N', 'D'),
        ('gamma', gamma): (1, 'Q'),
    })
    :param declarations: dict with (name, tensor) keys and (size_0, size_1, ...) values.
    size_i:
        - values of type int are checked directly.
        - values of other types act as symbols for lookups against declarations made by other tensors
    """
    # (symbol) => (declaring tensor name, declaring tensor dim, declared value)
    shape_declarations = {}
    for name, tensor in declarations.keys():
        intended_shape_symbols = declarations[(name, tensor)]
        actual_shape = _tensor_shape(tensor)

        has_same_num_tensor_dim = _num_dim(actual_shape) == _num_dim(intended_shape_symbols)

        assert has_same_num_tensor_dim, MESSAGE_NOT_SAME_NUM_TENSOR_DIMS_AS_DECLARED % (
            name, len(intended_shape_symbols), len(actual_shape)
        )

        for i, symbol in _enumerate_shape(intended_shape_symbols):
            actual_dim_value = actual_shape[i]

            if isinstance(symbol, int):
                declared_dim_value = symbol
                assert declared_dim_value == actual_dim_value, MESSAGE_NOT_CORRECT_STATIC_DIM_SIZE % (
                    name, i, actual_dim_value, declared_dim_value
                )
            elif symbol in shape_declarations:
                declarator_name, declarator_dim, declared_dim_value = shape_declarations[symbol]
                assert declared_dim_value == actual_dim_value, MESSAGE_NOT_SAME_DIM_SIZE_AS_DECLARED_BY % (
                    name, i, actual_dim_value, declared_dim_value, declarator_name, declarator_dim
                )
            else:
                shape_declarations[symbol] = (name, i, actual_shape[i])


def _tensor_shape(tensor):
    return tensor.get_shape().as_list()


def _num_dim(shape):
    if isinstance(shape, collections.Iterable):
        return len(shape)
    else:
        return 1


def _enumerate_shape(shape):
    if isinstance(shape, collections.Iterable):
        return enumerate(shape)
    else:
        return [(0, shape)]