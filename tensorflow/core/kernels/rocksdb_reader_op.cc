/*
 * Copyright (C) 2021  Lee, Conley <conleylee@foxmail.com>
 * Author: Lee, Conley <conleylee@foxmail.com>
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

#include "data/common.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class RocksdbReader : public ReaderBase {
 public:
  RocksdbReader(const string& node_name, std::string const& path_to_db,
                uint32_t const start, Env* env)
      : ReaderBase(strings::StrCat("RocksdbReader '", node_name, "'")),
        path_(path_to_db),
        start_(start),
        db_(nullptr),
        it_(nullptr),
        env_(env) {}

  Status OnWorkStartedLocked() override {
    rocksdb::DB* db = nullptr;
    auto status = open_db(path_.c_str(), &db, opt_);
    if (!status.ok()) {
      LOG(INFO) << "open db " << path_ << " failed: " << status.ToString();
      return Status::OK();
    }

    db_ = std::shared_ptr<rocksdb::DB>(db);
    auto it = db_->NewIterator(rocksdb::ReadOptions());

    std::string key = get_start_key_();
    it->Seek(key);

    if (!it->Valid()) {
      LOG(INFO) << "iterator is invalid: " << it->status().ToString();
    }
    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    it_.reset();
    db_.reset();
    return Status::OK();
  }

  Status ReadLocked(tstring* key, tstring* value, bool* produced,
                    bool* at_end) override {
    if (!it_->Valid()) {
      *at_end = true;
      return Status::OK();
    }

    *value = it_->value().ToString();
    *key = it_->key().ToString(true);
    *produced = true;

    it_->Next();
    return Status::OK();
  }

  Status ResetLocked() override {
    std::string key = get_start_key_();
    it_->Seek(key);

    return ReaderBase::ResetLocked();
  }

 private:
  std::string get_start_key_() {
    std::string key;
    const char* p = reinterpret_cast<const char*>(&start_);
    for (int i = sizeof(start_) - 1; i >= 0; --i) {
      key.push_back(p[i]);
    }

    return std::move(key);
  }

  std::string const path_;
  uint32_t const start_;
  std::shared_ptr<rocksdb::DB> db_;
  std::shared_ptr<rocksdb::Iterator> it_;
  rocksdb::Options opt_;
  Env* env_;
};

class RocksdbReaderOp : public ReaderOpKernel {
 public:
  explicit RocksdbReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    int start = -1;
    OP_REQUIRES_OK(context, context->GetAttr("start", &start));
    OP_REQUIRES(context, start >= 0,
                errors::InvalidArgument("start muse be >= 0 not ", start));
    std::string db;
    OP_REQUIRES_OK(context, context->GetAttr("db", &db));
    OP_REQUIRES(context, !db.empty(),
                errors::InvalidArgument("db must not be empty"));
    Env* env = context->env();
    SetReaderFactory([this, start, &db, env]() {
      return new RocksdbReader(name(), db, static_cast<uint32_t>(start), env);
    });
  }
};

}  // namespace tensorflow
