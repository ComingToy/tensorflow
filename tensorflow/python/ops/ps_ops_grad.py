from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


@ops.RegisterGradient("LookupEmbeddingLocalPsOp")
def lookup_embedding_local_ps_grad(op, grad):
    handle = op.inputs[0]
    ids = op.inputs[1]
    if isinstance(grad, ops.IndexedSlices):
        ids = array_ops.gather(ids, grad.indices)
        grad = grad.values

    shape = [ids.shape.as_list()[-1], grad.shape.as_list()[-1]]
    return [ops.IndexedSlices(grad, ids, shape), None]
