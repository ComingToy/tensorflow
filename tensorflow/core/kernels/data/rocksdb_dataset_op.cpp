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

#include <memory>

#include "common.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

namespace data {
constexpr char kCurrentExampleKey[] = "current_example_key";
class RocksdbDatasetOp : public DatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "Rocksdb";
  static constexpr const char* const kExampleDB = "example_db";
  static constexpr const char* const kStart = "start";

  explicit RocksdbDatasetOp(OpKernelConstruction* ctx);

 private:
  class Dataset;

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;
};

class RocksdbDatasetOp::Dataset : public DatasetBase {
 public:
  template <typename String>
  Dataset(OpKernelContext* ctx, String&& example_db, uint64 const start)
      : DatasetBase(DatasetContext(ctx)),
        example_db_path_(std::move(example_db)),
        start_(start) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      std::string const& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this,
        name_utils::IteratorPrefix(RocksdbDatasetOp::kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
    static DataTypeVector dtypes({DT_UINT64, DT_STRING});
    return dtypes;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    static std::vector<PartialTensorShape> shapes({{}, {}});
    return shapes;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    return Status::OK();
  }

  Status CheckExternalState() const override { return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* example_db = nullptr;
    Node* start = nullptr;

    tstring example_db_path = example_db_path_;

    TF_RETURN_IF_ERROR(b->AddScalar(example_db_path, &example_db));
    TF_RETURN_IF_ERROR(b->AddScalar(start_, &start));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {example_db, start}, output));

    return Status::OK();
  }

 private:
  std::string example_db_path_;
  uint64 const start_;

  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params) : DatasetIterator<Dataset>(params) {
      auto dataset = this->dataset();
      rocksdb::DB* example_db = nullptr;
      auto status =
          open_db(dataset->example_db_path_.c_str(), &example_db, example_opt_);
      if (!status.ok()) {
        LOG(INFO) << "open db " << dataset->example_db_path_
                  << " failed: " << status.ToString();
      }

      auto const start = static_cast<uint64>(dataset->start_);
      keybuf(start, cur_key_);

      example_db_ = std::shared_ptr<rocksdb::DB>(example_db);
      if (example_db_) {
        auto it = example_db_->NewIterator(rocksdb::ReadOptions());
        it_ = std::shared_ptr<rocksdb::Iterator>(it);
        it_->Seek(rocksdb::Slice(cur_key_));
      } else {
        it_ = nullptr;
      }
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      if (!example_db_ || !it_->Valid()) {
        *end_of_sequence = true;
        if (!it_->status().ok()) {
          LOG(INFO) << "iterate db failed: " << it_->status().ToString();
        }
        return Status::OK();
      }

      out_tensors->emplace_back(ctx->allocator({}), DT_UINT64, TensorShape({}));

      uint64_t key = 0;
      bufkey(it_->key().ToString(), key);

      out_tensors->back().scalar<uint64>()() = key;

      out_tensors->emplace_back(ctx->allocator({}), DT_STRING, TensorShape({}));
      out_tensors->back().scalar<tstring>()() =
          std::move(it_->value().ToString());

      cur_key_ = it_->key().ToString();
      it_->Next();
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kCurrentExampleKey), cur_key_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      tstring key;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kCurrentExampleKey), &key));
      cur_key_ = key.c_str();
      return Status::OK();
    }

   private:
    static void keybuf(uint64_t const& key, std::string& buf) {
      buf.clear();
      const char* p = reinterpret_cast<const char*>(&key);
      for (int i = sizeof(key) - 1; i >= 0; --i) {
        buf.push_back(p[i]);
      }
    }

    static void bufkey(std::string const& buf, uint64_t& key) {
      assert(buf.size() == sizeof(key));
      char* p = reinterpret_cast<char*>(&key);
      for (int i = sizeof(key) - 1; i >= 0; --i) {
        p[i] = buf[sizeof(key) - i - 1];
      }
    }

    std::shared_ptr<rocksdb::DB> example_db_;
    rocksdb::Options example_opt_;
    std::string cur_key_;
    std::shared_ptr<rocksdb::Iterator> it_;
  };
};

RocksdbDatasetOp::RocksdbDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {}

void RocksdbDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  const Tensor* example_db = nullptr;
  const Tensor* start = nullptr;

  OP_REQUIRES_OK(ctx, ctx->input(kExampleDB, &example_db));
  OP_REQUIRES_OK(ctx, ctx->input(kStart, &start));

  OP_REQUIRES(ctx, example_db->dims() == 0,
              errors::InvalidArgument("`example_db` must be a scalar"));
  OP_REQUIRES(ctx, start->dims() == 0,
              errors::InvalidArgument("`start` must be a scalar"));

  std::string path_to_example_db = example_db->scalar<tstring>()(0);
  auto start_example_id = start->scalar<uint64>()(0);

  *output = new Dataset(ctx, std::move(path_to_example_db), start_example_id);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("RocksdbDataset").Device(DEVICE_CPU),
                        RocksdbDatasetOp);
}
}  // namespace data
}  // namespace tensorflow
