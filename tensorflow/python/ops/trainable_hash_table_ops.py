from tensorflow.python.ops import gen_trainable_hash_table_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import dtypes


class TrainableHashTableVariable(object):
    def __init__(self, init_values, container, name, dims, trainable=True):
        self._handle = gen_trainable_hash_table_ops.local_ps_table_handle_op(
            init_values, container, name, name=name, dims=dims)
        self._container = container
        self._name = name
        self._dims = dims
        self._trainable = trainable
        self._dtype = dtypes.as_dtype('float32')
        self._shape = self._handle.shape

        if trainable:
            ops.add_to_collection(ops.GraphKeys.TRAINABLE_VARIABLES, self)
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS,
                              TrainableHashTableVariable._Saveable(self, name))

    @property
    def handle(self):
        return self._handle

    @property
    def dtype(self):
        return self._dtype

    @property
    def trainable(self):
        return self._trainable

    @property
    def name(self):
        return self._name

    @property
    def op(self):
        return self._handle.op

    @property
    def device(self):
        return self._handle.device

    def get_shape(self):
        return self._shape

    def export(self, name=None):
        return gen_trainable_hash_table_ops.local_ps_table_export_op(self.handle, name=name)

    def __repr__(self):
        return "<tf.TrainableHashTableVariable 'container=%s name=%s' shape=%s dtype=%s>" % (self._container, self.name, self.get_shape(), self.dtype.name)

    class _Saveable(BaseSaverBuilder.SaveableObject):
        def __init__(self, table, name):
            handle = table.export()
            specs = [
                BaseSaverBuilder.SaveSpec(handle, "", name + "-handle"),
            ]
            # pylint: disable=protected-access
            super(TrainableHashTableVariable._Saveable,
                  self).__init__(table, specs, name)

        def restore(self, restored_tensors, unused_restored_shape):
            return gen_trainable_hash_table_ops.local_ps_table_import_op(self.op._handle)


def embedding_lookup(params, ids, name=None):
    if params is None:
        raise ValueError('params must be specified')
    if isinstance(params, (list, tuple)) and len(params) != 1:
        raise ValueError('Only one params is allowed')
    if isinstance(params, (list, tuple)):
        params = params[0]
    if not isinstance(params, TrainableHashTableVariable):
        raise TypeError('type of params must be TrainableHashTableVariable')

    with ops.name_scope(name, 'embedding_lookup', [params, ids]) as name:
        if not isinstance(ids, ops.Tensor):
            ids = ops.convert_n_to_tensor(ids, name='ids')
        if ids.dtype != dtypes.string and ids.dtype != dtypes.int64:
            raise TypeError('dtype of ids must be string or int64')

        with ops.colocate_with(None, ignore_existing=True):
            flat_keys = array_ops.reshape(ids, (-1, ))
            embedding = gen_trainable_hash_table_ops.lookup_embedding_local_ps_op(
                params.handle, flat_keys)
            indices = math_ops.range(array_ops.size(flat_keys), dtype=dtypes.int64)
            indices = array_ops.reshape(indices, array_ops.shape(ids))
            embedding = array_ops.gather(embedding, indices)
            return embedding


def embedding_lookup_sparse(
        params, ids, weights, combiner='mean', name=None):
    if combiner not in ('mean', 'sqrtn', 'sum'):
        raise ValueError('combiner must be mean, sqrtn or sum')

    if not isinstance(ids, sparse_tensor.SparseTensor):
        raise TypeError('ids must be SparseTensor')

    if not isinstance(weights, sparse_tensor.SparseTensor):
        raise TypeError('weights must be SparseTensor')

    if weights.dtype != params.dtype:
        raise TypeError('weights.dtype != params.dtype')

    weights = weights.values
    with ops.name_scope(name, 'embedding_lookup_sparse', [ids, weights, params]) as name:
        segment_ids = ids.indices[:, 0]
        if segment_ids.dtype != dtypes.int64:
            segment_ids = math_ops.cast(segment_ids, dtypes.int64)

        ids = ids.values
        ids, idx = array_ops.unique(ids)

        embeddings = embedding_lookup(
            params, ids, name=name + '/embedding_lookup_sparse')

        embeddings = array_ops.gather(embeddings, idx)

        ones = array_ops.fill(array_ops.expand_dims(
            array_ops.rank(embeddings) - 1, 0), 1)

        bcast_weights_shape = array_ops.concat(
            [array_ops.shape(weights), ones], 0)

        orig_weights_shape = weights.get_shape()
        weights = array_ops.reshape(weights, bcast_weights_shape)

        if embeddings.get_shape().ndims is not None:
            weights.set_shape(
                orig_weights_shape.concatenate(
                    [1 for _ in range(embeddings.get_shape().ndims - 1)]))

        embeddings *= weights

        if combiner == "sum":
            embeddings = math_ops.segment_sum(
                embeddings, segment_ids, name=name)
        elif combiner == "mean":
            embeddings = math_ops.segment_sum(embeddings, segment_ids)
            weight_sum = math_ops.segment_sum(weights, segment_ids)
            embeddings = math_ops.div(embeddings, weight_sum, name=name)
        elif combiner == "sqrtn":
            embeddings = math_ops.segment_sum(embeddings, segment_ids)
            weights_squared = math_ops.pow(weights, 2)
            weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
            weight_sum_sqrt = math_ops.sqrt(weight_sum)
            embeddings = math_ops.div(embeddings, weight_sum_sqrt, name=name)
        else:
            assert False, "Unrecognized combiner"

        return embeddings
