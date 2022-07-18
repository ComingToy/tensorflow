/*
 * Copyright (C) 2022  Conley Lee <conleylee@foxmail.com>
 * Author: Conley Lee <conleylee@foxmail.com>
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

#include <errno.h>
#include <stdlib.h>
#include <unistd.h>

#include <fstream>
#include <functional>
#include <iterator>
#include <unordered_set>

#include "comm_feats_generated.h"
#include "common.h"
#include "example_generated.h"
#include "feature_generated.h"
#include "tensor_utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

class AliCCPRocksDBOp : public OpKernel {
 public:
  explicit AliCCPRocksDBOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("max_feats", &max_feats_));
  }

  void parse_examples(OpKernelContext* context,
                      std::vector<const dataset::Example*> const& examples,
                      Tensor* field_id_tensor, Tensor* feat_id_tensor,
                      Tensor* feats_tensor, Tensor* y, Tensor* z,
                      Tensor* lens_tensor) {
    auto feat_matrix = feats_tensor->matrix<float>();
    auto field_id_matrix = field_id_tensor->matrix<int64>();
    auto feat_id_matrix = feat_id_tensor->matrix<int64>();

    auto batch_size = 64;
    auto batch_nums = (examples.size() + batch_size - 1) / batch_size;

    auto parse_batch = [this, batch_size, examples, &field_id_matrix,
                        &feat_matrix, &feat_id_matrix, &y, &z,
                        &lens_tensor](Eigen::Index start, Eigen::Index end) {
      start = std::min(start * batch_size, (Eigen::Index)examples.size());
      end = std::min(end * batch_size, (Eigen::Index)examples.size());

      for (auto i = start; i < end; ++i) {
        auto const example = examples[i];
        if (!example) {
          continue;
        }

        y->flat<int64>()(i) = static_cast<int64>(example->y());
        z->flat<int64>()(i) = static_cast<int64>(example->z());

        auto feats = example->feats();

        int k = 0;
        int len = std::min((int32)feats->Length(), max_feats_);
        for (; k < len; ++k) {
          auto feat = feats->Get(k);
          auto field_id = feat->feat_field_id();
          auto feat_id = feat->feat_id();
          auto value = feat->value();

          feat_matrix(i, k) = value;
          field_id_matrix(i, k) = static_cast<int64>(field_id);
          feat_id_matrix(i, k) = static_cast<int64>(feat_id);
        }

        if (!example->comm_feat_id() &&
            example->comm_feat_id()->str().empty()) {
          continue;
        }

        lens_tensor->flat<int64>()(i) = k;
        for (; k < max_feats_; ++k) {
          feat_matrix(i, k) = 0.0;
          field_id_matrix(i, k) = 0;
          feat_id_matrix(i, k) = 0;
        }
      }
    };

    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    auto cost_per_unit = 10 * 6000 * batch_size;

    thread_pool->ParallelFor(batch_nums, cost_per_unit, std::move(parse_batch));
  }

  void Compute(OpKernelContext* context) override {
    auto const input = context->input(0);

    OP_REQUIRES(context, input.dims() == 1,
                Status(error::INVALID_ARGUMENT, "1d tensor is accepted only."));

    auto const nelems = static_cast<int32>(input.NumElements());

    auto field_id_tensor = alloc_tensor(context, {nelems, max_feats_}, 0);
    auto feat_id_tensor = alloc_tensor(context, {nelems, max_feats_}, 1);
    auto feats_tensor = alloc_tensor(context, {nelems, max_feats_}, 2);
    auto y = alloc_tensor(context, {nelems}, 3);
    auto z = alloc_tensor(context, {nelems}, 4);
    auto lens_tensor = alloc_tensor(context, {nelems}, 5);

    if (!field_id_tensor || !feat_id_tensor || !feats_tensor || !y || !z ||
        !lens_tensor) {
      return;
    }

    std::vector<const dataset::Example*> examples;
	auto feats = input.vec<tstring>();
	
	for (size_t i = 0; i < feats.size(); ++i){
		auto const& buf = feats(i);
		examples.push_back(dataset::GetExample(buf.data()));
	}

    parse_examples(context, examples, field_id_tensor, feat_id_tensor,
                   feats_tensor, y, z, lens_tensor);
  }

 private:
  int32 max_feats_;
};

REGISTER_KERNEL_BUILDER(Name("AliCCPRocksDB").Device(DEVICE_CPU),
                        AliCCPRocksDBOp);
};  // namespace tensorflow

