#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <mutex>

#include "Eigen/Dense"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/tools/dataset/embedding_generated.h"

namespace tensorflow {

class __LocalPsTableInterface : public ResourceBase {
 public:
  virtual Status read(Tensor const& keys, Tensor& values,
                      const bool create = true) = 0;
  virtual Status update(Tensor const& keys, Tensor const& values) = 0;
  virtual Status export_values(int64* counts) = 0;
  virtual Status import_values(int64* counts) = 0;

  std::string const& container() const { return container_; }
  std::string const& name() const { return name_; }
  int dims() const { return dims_; }

  __LocalPsTableInterface(std::string const& container, std::string const& name,
                          int const dims)
      : container_(container), name_(name), dims_(dims) {}

 private:
  const std::string container_;
  const std::string name_;
  const int dims_;
};

class __LocalMemPsTable : public __LocalPsTableInterface {
 public:
  __LocalMemPsTable(OpKernelContext* context, std::string const& container,
                    std::string const& name, int const dims,
                    Tensor const& init_values)
      : __LocalPsTableInterface(container, name, dims) {
    OP_REQUIRES(context, init_values.dims() == 2,
                Status(error::INVALID_ARGUMENT, "init_values must be 2 ranks"));

    auto init_values_dim0 = init_values.dim_size(0);
    auto init_values_dim1 = init_values.dim_size(1);
    auto shaped_init_values =
        init_values.shaped<float, 2>({init_values_dim0, init_values_dim1});

    init_values_.resize(init_values_dim0);
    for (int i = 0; i < init_values_dim0; ++i) {
      auto& vec = init_values_[i];
      for (int k = 0; k < init_values_dim1; ++k) {
        vec.push_back(shaped_init_values(i, k));
      }
    }

    if (::access(container.c_str(), F_OK) != 0) {
      int ret = ::mkdir(container.c_str(), S_IWUSR | S_IRUSR | S_IXUSR);
      char buf[1024];
      OP_REQUIRES(context, ret == 0,
                  errors::Internal("mkdir ", container, " failed: ",
                                   strerror_r(errno, buf, sizeof(buf))));
      LOG(INFO) << "create embedding checkpoint dir " << container;
    }

    auto checkpoint_path = container + "/" + name;
    if (::access(checkpoint_path.c_str(), F_OK) != 0) {
      int ret = ::mkdir(checkpoint_path.c_str(), S_IWUSR | S_IRUSR | S_IXUSR);
      char buf[1024];
      OP_REQUIRES(context, ret == 0,
                  errors::Internal("mkdir ", checkpoint_path, " failed: ",
                                   strerror_r(errno, buf, sizeof(buf))));
      LOG(INFO) << "create embedding checkpoint dir " << checkpoint_path;
    }

    // int64 counts = 0;
    // OP_REQUIRES_OK(context, import_values(&counts));
  }

  std::string DebugString() const override { return "__LocalMemPsTable"; }

  Status read(Tensor const& keys, Tensor& values, bool const create) override {
    auto keys_rank = keys.dims();
    auto values_rank = values.dims();
    if (keys_rank != 1) {
      return errors::InvalidArgument("keys rank must be 1. ", keys_rank,
                                     " is invalid");
    }

    if (values_rank != 2) {
      return errors::InvalidArgument("values rank must be 2. ", values_rank,
                                     " is invalid");
    }

    if (keys.dim_size(0) != values.dim_size(0)) {
      return errors::InvalidArgument(
          "keys.dims_size(0) != values.dim_size(0), keys.dims_size(0) = ",
          keys.dim_size(0), ", values.dim_size(0) = ", values.dim_size(0));
    }

    if (values.dim_size(1) != dims()) {
      return errors::InvalidArgument(
          "values.dim_size(1) != dims, values.dim_size(1) = ",
          values.dim_size(1), ", dims = ", dims());
    }

    {
      std::lock_guard<std::mutex> ml(mu_);
      auto flat_keys = keys.flat<int64>();
      auto flat_values = values.shaped<float, 2>({values.dim_size(0), dims()});

      for (int i = 0; i < keys.NumElements(); ++i) {
        if (table_.count(flat_keys(i)) == 0 && !create) {
          return errors::NotFound("key not found: ", flat_keys(i));
        }

        auto& vec = table_[flat_keys(i)];
        if (vec.empty()) {
          auto init_idx =
              random() % static_cast<unsigned long>(init_values_.size());
          vec = init_values_[init_idx];
        }

        for (int k = 0; k < dims(); ++k) {
          flat_values(i, k) = vec[k];
        }
      }
    }

    return Status::OK();
  }

  Status update(Tensor const& keys, Tensor const& values) override {
    auto keys_rank = keys.dims();
    auto values_rank = values.dims();
    if (keys_rank != 1) {
      return errors::InvalidArgument("keys rank must be 1. ", keys_rank,
                                     " is invalid");
    }

    if (values_rank != 2) {
      return errors::InvalidArgument("values rank must be 2. ", values_rank,
                                     " is invalid");
    }

    if (keys.dim_size(0) != values.dim_size(0)) {
      return errors::InvalidArgument(
          "keys.dims_size(0) != values.dim_size(0), keys.dims_size(0) = ",
          keys.dim_size(0), ", values.dim_size(0) = ", values.dim_size(0));
    }

    if (values.dim_size(1) != dims()) {
      return errors::InvalidArgument(
          "values.dim_size(1) != dims, values.dim_size(1) = ",
          values.dim_size(1), ", dims = ", dims());
    }

    {
      std::lock_guard<std::mutex> ml(mu_);
      auto flat_keys = keys.flat<int64>();

      auto n = static_cast<int>(keys.NumElements());
      auto d = dims();

      auto shaped_values = values.shaped<float, 2>({n, d});

      for (int i = 0; i < keys.NumElements(); ++i) {
        // check exist
        auto pos = table_.find(flat_keys(i));
        if (pos == table_.end()) {
          return errors::InvalidArgument("try to update key not in table: ",
                                         flat_keys(i));
        }
        auto& vec = pos->second;
        vec.resize(static_cast<size_t>(d));
        for (int k = 0; k < d; ++k) {
          auto elem = shaped_values(i, k);
          vec[k] = elem;
        }
      }
    }

    return Status::OK();
  }

  Status export_values(int64* counts) override {
    ::flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<::embedding::KVEntry>> kvs;

    {
      std::lock_guard<std::mutex> ml(mu_);
      for (const auto& it : table_) {
        kvs.emplace_back(
            ::embedding::CreateKVEntryDirect(builder, it.first, &it.second));
      }

      *counts = static_cast<int64>(table_.size());

      auto embs = ::embedding::CreateEmbeddingDirect(builder, &kvs);
      builder.Finish(embs);

      auto checkpoint_file = container() + "/" + name() + "/values.bin";
      auto checkpoint_file_tmp = container() + "/" + name() + "/values.bin.tmp";
      auto fout = ::fopen(checkpoint_file_tmp.c_str(), "wb+");
      if (!fout) {
        char buf[1024];
        return errors::Internal("failed at opening checkpoint file ",
                                checkpoint_file_tmp, ", ",
                                strerror_r(errno, buf, sizeof(buf)));
      }

      const auto* p = builder.GetBufferPointer();
      auto const size = builder.GetSize();
      size_t n = 0;
      while (n < size) {
        auto ret = fwrite(p, sizeof(uint8_t), size, fout);
        if (ret == 0) {
          char buf[1024];
          return errors::Internal("failed at writing checkpoint file ",
                                  checkpoint_file_tmp, ", ",
                                  strerror_r(errno, buf, sizeof(buf)));
        }
        n += ret;
      }

      auto ret = ::fflush(fout);
      if (ret != 0) {
        char buf[1024];
        return errors::Internal("failed at flushing checkpoint file ",
                                checkpoint_file_tmp, ", ",
                                strerror_r(errno, buf, sizeof(buf)));
      }

      ret = fclose(fout);
      if (ret != 0) {
        char buf[1024];
        return errors::Internal("failed at closing checkpoint file ",
                                checkpoint_file_tmp, ", ",
                                strerror_r(errno, buf, sizeof(buf)));
      }

      ret = ::rename(checkpoint_file_tmp.c_str(), checkpoint_file.c_str());
      if (ret != 0) {
        char buf[1024];
        return errors::Internal("failed at rename checkpoint file ",
                                checkpoint_file_tmp, ", ",
                                strerror_r(errno, buf, sizeof(buf)));
      }

      LOG(INFO) << "flush embedding container = " << container()
                << ", name = " << name() << " to file " << checkpoint_file
                << ", size = " << table_.size();
    }
    return Status::OK();
  }

  Status import_values(int64* counts) override {
    auto checkpoint_file = container() + "/" + name() + "/values.bin";
    if (::access(checkpoint_file.c_str(), F_OK) != 0) {
      LOG(INFO) << checkpoint_file
                << " is not exists. continue with init values";
      return Status::OK();
    }

    std::ifstream ifs(checkpoint_file, std::ios::binary | std::ios::in);
    if (!ifs.good()) {
      return errors::Internal("cannot open and read checkpoint file ",
                              checkpoint_file);
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
    auto const* emb = ::embedding::GetEmbedding(content.c_str());
    if (!emb || !emb->entries()) {
      return errors::Internal(
          "failed at parsing embedding from checkpoint file ", checkpoint_file);
    }

    {
      std::lock_guard<std::mutex> ml(mu_);
      for (auto const& kv : *emb->entries()) {
        if (!kv) continue;
        const auto k = kv->key();
        const auto v = kv->values();

        table_[k] = std::vector<float>(v->begin(), v->end());
      }

      LOG(INFO) << "restore embedding container = " << container()
                << ", name = " << name() << " from checkpoint "
                << checkpoint_file << " number of keys "
                << emb->entries()->Length();

      *counts = static_cast<int64>(table_.size());
    }
    return Status::OK();
  }

 private:
  std::vector<std::vector<float>> init_values_;
  std::unordered_map<int64, std::vector<float>> table_;
  mutable std::mutex mu_;
};

class LocalPsTableHandleOp : public OpKernel {
 public:
  LocalPsTableHandleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_name", &name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dims", &dims_));

    is_anonymous_ = name_ == ResourceHandle::ANONYMOUS_NAME;
    assert(!is_anonymous_);

    if (!is_anonymous_) {
      AllocatorAttributes attr;
      attr.set_on_host(true);
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DT_RESOURCE, {}, &resource_, attr));
      auto handle =
          MakeResourceHandle<__LocalPsTableInterface>(ctx, container_, name_);
      resource_.scalar<ResourceHandle>()() = handle;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    auto init_values = ctx->input(0);
    auto init_values_ranks = init_values.dims();
    OP_REQUIRES(ctx, init_values_ranks == 2,
                Status(error::INVALID_ARGUMENT, "init_values must be 2 ranks"));
    auto init_values_dim0 = init_values.dim_size(0);
    auto init_values_dim1 = init_values.dim_size(1);
    auto shaped_init_values =
        init_values.shaped<float, 2>({init_values_dim0, init_values_dim1});

    core::RefCountPtr<__LocalPsTableInterface> v;
    OP_REQUIRES_OK(
        ctx, LookupOrCreateResource<__LocalPsTableInterface>(
                 ctx, resource_.scalar<ResourceHandle>()(), &v,
                 [this, &init_values, &ctx](__LocalPsTableInterface** ptr) {
                   *ptr = new __LocalMemPsTable(ctx, container_, name_, dims_,
                                                init_values);

                   LOG(INFO)
                       << "create LocalPsTableHandleOp instance, container = "
                       << container_ << ", name = " << name_
                       << ", dims = " << (*ptr)->dims();
                   return Status::OK();
                 }));
    ctx->set_output(0, resource_);
  }

 private:
  Tensor resource_;
  bool is_anonymous_;
  std::string name_;
  std::string container_;
  int dims_;
};

class LocalPsTableExportOp : public OpKernel {
 public:
  LocalPsTableExportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<__LocalPsTableInterface> v;
    int64 counts = 0;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &v));
    OP_REQUIRES_OK(ctx, v->export_values(&counts));

    Tensor* output;
    ctx->allocate_output(0, {}, &output);
    output->scalar<int64>()() = counts;
  }
};

class LocalPsTableImportOp : public OpKernel {
 public:
  LocalPsTableImportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<__LocalPsTableInterface> v;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &v));

    int64 counts = 0;
    OP_REQUIRES_OK(ctx, v->import_values(&counts));
    Tensor* output;
    ctx->allocate_output(0, {}, &output);
    output->scalar<int64>()() = counts;
  };
};
class LookupEmbeddingLocalPsOp : public OpKernel {
 public:
  LookupEmbeddingLocalPsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    core::RefCountPtr<__LocalPsTableInterface> v;
    OP_REQUIRES_OK(context,
                   LookupResource(context, HandleFromInput(context, 0), &v));

    auto ids = context->input(1);

    OP_REQUIRES(context, ids.dims() == 1,
                errors::InvalidArgument("rank of ids is invalid: ", ids.dims(),
                                        ", only rank 1 is supported"));

    TensorShape ids_shape = ids.shape();
    auto n = ids_shape.dim_size(0);
    Tensor* output = nullptr;

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {n, v->dims()}, &output));
    OP_REQUIRES_OK(context, v->read(ids, *output));

    return;
  }
};

struct __ScatterAddOp {
  float operator()(float const lhs, float const rhs) const { return lhs + rhs; }
};

struct __ScatterAssignOp {
  float operator()(float const lhs, float const rhs) const { return rhs; }
};

struct __ScatterSubOp {
  float operator()(float const lhs, float const rhs) const { return lhs - rhs; }
};

struct __ScatterMulOp {
  float operator()(float const lhs, float const rhs) const { return lhs * rhs; }
};

struct __ScatterDivOp {
  float operator()(float const lhs, float const rhs) const { return lhs / rhs; }
};

template <typename __OP>
class ScatterEmbeddingLocalPsOp : public OpKernel {
 public:
  ScatterEmbeddingLocalPsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<__LocalPsTableInterface> v;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &v));

    auto ids = ctx->input(1);
    auto values = ctx->input(2);

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, values.shape(), &output));
    auto& old_values = *output;

    OP_REQUIRES_OK(ctx, v->read(ids, old_values, false));

    // do update
    int n = static_cast<int>(ids.NumElements());
    auto shaped_old_values = old_values.shaped<float, 2>({n, v->dims()});
    auto shaped_values = values.shaped<float, 2>({n, v->dims()});

    __OP op;
    for (auto i = 0; i < ids.NumElements(); ++i) {
      for (auto k = 0; k < v->dims(); ++k) {
        shaped_old_values(i, k) =
            op(shaped_old_values(i, k), shaped_values(i, k));
      }
    }

    OP_REQUIRES_OK(ctx, v->update(ids, old_values));
  }
};

REGISTER_KERNEL_BUILDER(Name("LocalPsTableHandleOp").Device(DEVICE_CPU),
                        LocalPsTableHandleOp);
REGISTER_KERNEL_BUILDER(Name("LookupEmbeddingLocalPsOp")
                            .Device(DEVICE_CPU)
                            .HostMemory("ps_handle")
                            .HostMemory("ids"),
                        LookupEmbeddingLocalPsOp);
REGISTER_KERNEL_BUILDER(Name("LocalPsTableExportOp").Device(DEVICE_CPU),
                        LocalPsTableExportOp);
REGISTER_KERNEL_BUILDER(Name("LocalPsTableImportOp").Device(DEVICE_CPU),
                        LocalPsTableImportOp);

#define REGISTER_PS_KERNEL(__OP)                                     \
  REGISTER_KERNEL_BUILDER(Name("Scatter" #__OP "EmbeddingLocalPsOp") \
                              .Device(DEVICE_CPU)                    \
                              .HostMemory("ps_handle")               \
                              .HostMemory("ids")                     \
                              .HostMemory("values"),                 \
                          ScatterEmbeddingLocalPsOp<__Scatter##__OP##Op>)

REGISTER_PS_KERNEL(Assign);
REGISTER_PS_KERNEL(Add);
REGISTER_PS_KERNEL(Sub);
REGISTER_PS_KERNEL(Mul);
REGISTER_PS_KERNEL(Div);

#undef REGISTER_PS_KERNEL

};  // namespace tensorflow
