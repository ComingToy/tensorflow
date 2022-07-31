#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("PslitePushPull")
    .Attr("op: {'Push', 'Pull', 'PushPull'}")
    .Attr("cmd: {'delta', 'overwrite'}")
    .Attr("var_name: string = ''")
    .Input("keys: int64")
    .Input("value: float32")
    .Output("updated: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("PsliteMyRank")
    .Output("rank: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });
REGISTER_OP("PsliteSyncGlobalStep")
    .Input("local_global_step: resource")
    .Output("global_step: int64")
    .Attr("op: {'Pull', 'PushPull'}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });
}  // namespace tensorflow
