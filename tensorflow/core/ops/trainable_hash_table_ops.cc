#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("LocalPsTableHandleOp")
    .Input("init_values: float32")
    .Output("handle: resource")
    .Attr("container: string")
    .Attr("table_name: string")
    .Attr("dims: int")
	.Attr("training: bool = false")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      int dims = 0;
      TF_RETURN_IF_ERROR(ctx, ctx->GetAttr("dims", &dims));
      if (dims <= 0) {
        return errors::InvalidArgument("dims <= 0 is invalid. dims = ", dims);
      }

      auto outshape = ctx->MakeShape({ctx->UnknownDim(), ctx->MakeDim(dims)});
      ctx->set_output(0, outshape);
      return Status::OK();
    });

REGISTER_OP("LocalPsTableExportOp")
    .Input("ps_handle: resource")
    .Output("counts: int64")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      ctx->set_output(0, ctx->Scalar());
      return Status::OK();
    });

REGISTER_OP("LocalPsTableImportOp")
    .Input("ps_handle: resource")
    .Output("counts: int64")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      ctx->set_output(0, ctx->Scalar());
      return Status::OK();
    });

REGISTER_OP("LookupEmbeddingLocalPsOp")
    .Input("ps_handle: resource")
    .Input("ids: int64")
    .Output("emb: float32")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      auto params_shape = ctx->input(0);
      auto ids_shape = ctx->input(1);

      auto out_shape =
          ctx->MakeShape({ctx->Dim(ids_shape, 0), ctx->Dim(params_shape, 1)});
      ctx->set_output(0, out_shape);

      return Status::OK();
    });

#define REGISTER_PS_OP(__OP)                                   \
  REGISTER_OP("Scatter" #__OP "EmbeddingLocalPsOp")            \
      .Input("ps_handle: resource")                            \
      .Input("ids: int64")                                     \
      .Input("values: float32")                                \
      .Output("output_values: float32")                        \
      .SetShapeFn([](shape_inference::InferenceContext* ctx) { \
        ctx->set_output(0, ctx->input(2));                     \
        return Status::OK();                                   \
      })

REGISTER_PS_OP(Assign);
REGISTER_PS_OP(Add);
REGISTER_PS_OP(Sub);
REGISTER_PS_OP(Mul);
REGISTER_PS_OP(Div);

#undef REGISTER_PS_OP
}  // namespace tensorflow
