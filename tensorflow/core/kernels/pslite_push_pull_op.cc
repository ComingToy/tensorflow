/*
 * Copyright (C) 2021  Lee, Conley <743703241@qq.com>
 * Author: Lee, Conley <743703241@qq.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <functional>
#include <string>

#include "Eigen/Dense"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/pslite_ops.h"

namespace tensorflow {
class PsliteMyRankOp : public OpKernel {
 public:
  explicit PsliteMyRankOp(OpKernelConstruction* context) : OpKernel(context) {}

  virtual void Compute(OpKernelContext* context) override {
    Tensor* output;
    TensorShape shape;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
    auto rank = ps_my_rank();
    output->flat<int32_t>().setConstant(rank);
  }
};

template <class Device>
class PsliteSyncGlobalStepOp : public OpKernel {
 public:
  explicit PsliteSyncGlobalStepOp(OpKernelConstruction* context)
      : OpKernel(context) {
    std::string op;
    OP_REQUIRES_OK(context, context->GetAttr("op", &op));

    if (op == "Pull") {
      op_ = Pull;
    } else {
      op_ = PushPull;
    }
  };

  virtual void Compute(OpKernelContext* context) override {
    std::hash<std::string> hash_fn;
    constexpr size_t n = sizeof(int64_t) / sizeof(float);
    std::vector<float> values(n);
    const std::vector<uint64_t> keys = {hash_fn("__global_steps__")};
    std::vector<int> lens = {n};
    std::vector<float> out;

    if (op_ == Pull) {
      ps_pull_data(keys, out, lens, Delta);
    } else {
      ps_push_pull_data(keys, values, out, lens, Delta);
    }

    OP_REQUIRES(context, !out.empty(),
                Status(error::INTERNAL, "push pull global steps failed."));

    auto const* buffer = reinterpret_cast<const int64*>(out.data());
    const int64 steps = *buffer;

    core::RefCountPtr<Var> variable;
    OP_REQUIRES_OK(context,
                   LookupOrCreateResource<Var>(
                       context, HandleFromInput(context, 0), &variable,
                       [steps](Var** var) {
                         *var = new Var(DT_INT64);
                         auto ptr = *var;
                         ptr->tensor()->flat<int64>().setConstant(steps);
                         ptr->is_initialized = true;
                         return Status::OK();
                       }));

    mutex_lock guard(*variable->mu());
    OP_REQUIRES(context, variable->tensor()->dtype() == DT_INT64,
                errors::InvalidArgument(
                    "Trying to assign with worng dtype. Expected int64 , got ",
                    DataTypeString(variable->tensor()->dtype())));

    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);

    if (variable->copy_on_read_mode.load()) {
      PersistentTensor unused;
      Tensor* tmp;
      OP_REQUIRES_OK(context, context->allocate_persistent(
                                  DT_INT64, {1}, &unused, &tmp, attr));
      tmp->flat<int64>().setConstant(steps);
      *variable->tensor() = *tmp;
    } else {
      variable->tensor()->flat<int64>().setConstant(steps);
    }

    Tensor* output_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {1}, &output_tensor, attr));
    output_tensor->flat<int64>().setConstant(steps);
  }

 private:
  OpType op_;
};

class PsliteOp : public OpKernel {
 public:
  explicit PsliteOp(OpKernelConstruction* context) : OpKernel(context) {
    std::string op;
    std::string cmd;
    OP_REQUIRES_OK(context, context->GetAttr("op", &op));
    OP_REQUIRES_OK(context, context->GetAttr("cmd", &cmd));
    OP_REQUIRES_OK(context, context->GetAttr("var_name", &_var_name));

    assert(op == "Push" || op == "Pull" || op == "PushPull");
    assert(cmd == "delta" || cmd == "overwrite");

    if (op == "Push") {
      _op = Push;
    } else if (op == "Pull") {
      _op = Pull;
    } else {
      _op = PushPull;
    }

    if (cmd == "delta") {
      _cmd = Delta;
    } else {
      _cmd = Overwrite;
    }
  }

  Status check_shape(Tensor const& keys, Tensor const& values) {
    char msg[2048];

    if (!(values.dims() > 0 && keys.dims() == 1)) {
      ::snprintf(msg, sizeof(msg),
                 "check dims error keys.dim() = %d, values.dims() = %d",
                 keys.dims(), values.dims());
      return Status(error::INVALID_ARGUMENT, msg);
    }

    if (!((values.dim_size(0) == keys.dim_size(0)) || keys.dim_size(0) == 1)) {
      ::snprintf(msg, sizeof(msg),
                 "check dim_size error keys.dim_size(0)= %ld, "
                 "values.dim_size(0) = %ld",
                 keys.dim_size(0), values.dim_size(0));

      return Status(error::INVALID_ARGUMENT, msg);
    }

    return Status::OK();
  }

  void gen_kvs(Tensor const* keys_tensor, Tensor const* values_tensor,
               std::vector<uint64_t>& ps_key, std::vector<float>& vals,
               std::vector<int>& lens, std::vector<int>& indices) {
    assert(keys_tensor);
    static std::hash<int64> hash_fn;
    auto keys = keys_tensor->flat<int64>();
    std::vector<uint64_t> hash_keys;
    for (auto i = 0; i < keys_tensor->NumElements(); ++i) {
      hash_keys.push_back(hash_fn(keys(i)));
    }

    indices.resize(keys_tensor->NumElements());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [hash_keys](int const lhs, int const rhs) {
                return hash_keys[lhs] < hash_keys[rhs];
              });

    if (keys.size() == 1) {
      ps_key.push_back(hash_keys[0]);
      if (values_tensor) {
        lens.push_back(values_tensor->NumElements());
        auto p = reinterpret_cast<const float*>(values_tensor->data());
        std::copy(p, p + values_tensor->NumElements(),
                  std::back_inserter(vals));
      }
    } else {
      for (auto i = 0; i < keys.size(); ++i) {
        ps_key.push_back(hash_keys[indices[i]]);
        if (values_tensor) {
          auto const slice = values_tensor->SubSlice(indices[i]);
          // deal with align
          auto p = reinterpret_cast<float*>(slice.data());
          lens.push_back(slice.NumElements());
          std::copy(p, p + slice.NumElements(), std::back_inserter(vals));
        }
      }
    }
  }

  virtual void Compute(OpKernelContext* context) override {
    Tensor const& keys_tensor = context->input(0);
    Tensor const& values_tensor = context->input(1);

    OP_REQUIRES_OK(context, check_shape(keys_tensor, values_tensor));

    std::vector<float> out;
    std::vector<uint64_t> ps_key;
    std::vector<float> vals;
    std::vector<int> indices;
    std::vector<int> lens;

    gen_kvs(&keys_tensor, &values_tensor, ps_key, vals, lens, indices);
    std::vector<int> bak_lens(lens.cbegin(), lens.cend());

    if (_op == PushPull) {
      ps_push_pull_data(ps_key, vals, out, lens, _cmd);
    } else if (_op == Push) {
      ps_push_data(ps_key, vals, lens, _cmd);
    } else {
      ps_pull_data(ps_key, out, lens, _cmd);
    }

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, values_tensor.shape(),
                                                     &output_tensor));
    auto pout = output_tensor->flat<float>();

    if (keys_tensor.NumElements() == 1) {
      assert(out.empty() || out.size() == values_tensor.NumElements());
      auto const buffer = out.empty()
                              ? reinterpret_cast<float*>(values_tensor.data())
                              : out.data();
      for (auto i = 0; i < values_tensor.NumElements(); ++i) {
        pout(i) = buffer[i];
      }

      if (out.empty() && _op != Push) {
        // create kv
        ps_push_data(ps_key, vals, bak_lens, Overwrite);
      }

      return;
    }

    size_t pos = 0;
    std::vector<size_t> offsets = {0};
    for (auto i = 1; i < lens.size() + 1; ++i) {
      offsets.push_back(offsets[i - 1] + lens[i - 1]);
    }

    std::set<int> miss_indices;
    std::vector<uint64_t> miss_keys;
    std::vector<int> miss_lens;
    std::vector<float> miss_vals;

    for (auto i = 0; i < keys_tensor.NumElements(); ++i) {
      auto const slice = values_tensor.SubSlice(i);
      auto const n = slice.NumElements();
      auto const offset = offsets[indices[i]];

      // TODO: push miss key again
      const float* p = reinterpret_cast<float*>(slice.data());
      if (_op != Push && lens[indices[i]] == n) {
        p = out.data() + offset;
      } else if (_op != Push) {
        miss_indices.insert(indices[i]);
      }

      for (auto k = 0; k < n; ++k) {
        pout(i * n + k) = p[k];
      }
      pos += lens[indices[i]];
    }

    if (!miss_indices.empty()) {
      auto pvals = vals.begin();
      for (auto i = 0; i < ps_key.size(); ++i) {
        if (miss_indices.count(i) > 0) {
          miss_keys.push_back(ps_key[i]);
          miss_lens.push_back(bak_lens[i]);
          std::copy(pvals, pvals + bak_lens[i], std::back_inserter(miss_vals));
        }
        pvals += bak_lens[i];
      }

      ps_push_data(miss_keys, miss_vals, miss_lens, Overwrite);
    }
  }

 private:
  OpType _op;
  CmdType _cmd;
  std::string _var_name;
};

REGISTER_KERNEL_BUILDER(Name("PslitePushPull").Device(DEVICE_CPU), PsliteOp);
REGISTER_KERNEL_BUILDER(Name("PsliteMyRank").Device(DEVICE_CPU),
                        PsliteMyRankOp);
REGISTER_KERNEL_BUILDER(Name("PsliteSyncGlobalStep").Device(DEVICE_CPU),
                        PsliteSyncGlobalStepOp<CPUDevice>);
}  // namespace tensorflow
