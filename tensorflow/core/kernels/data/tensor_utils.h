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

#ifndef __TENSOR_UTILS_H__
#define __TENSOR_UTILS_H__
#include "common.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
inline Tensor*
alloc_tensor(OpKernelContext* context, std::vector<int32> const& dims, int const i)
{
    TensorShape shape;
    auto status = TensorShapeUtils::MakeShape(dims.data(), dims.size(), &shape);
    if (!status.ok()) {
        context->CtxFailure(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT, status.ToString()));
        return nullptr;
    }

    Tensor* tensor = nullptr;
    status = context->allocate_output(i, shape, &tensor);
    if (!status.ok()) {
        context->CtxFailure(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT, status.ToString()));
        return nullptr;
    }

    return tensor;
}

template<typename T>
void to_key(T const k, char* key)
{
    const char* p = reinterpret_cast<const char*>(&k);
    for (int i = sizeof(T) - 1; i >= 0; --i){
        key[sizeof(T) - 1 - i] = p[i];
    }
}

template<typename InputType, typename KeyType>
::rocksdb::Status
read_values(std::shared_ptr<::rocksdb::DB> db,
            ::rocksdb::ReadOptions const& opt,
            Tensor const& input,
            std::vector<std::string>& values)
{
    auto nelems = input.NumElements();
    auto input_flat = input.flat<InputType>();
    std::vector<rocksdb::Slice> keys;

    auto* keybuf = (char*)::malloc(sizeof(KeyType) * nelems);
    if (!keybuf) { return ::rocksdb::Status::Aborted("malloc buf failed."); }

    for (auto i = 0; i < nelems; ++i) {
        auto example_id = static_cast<KeyType>(input_flat(i));
        to_key(example_id, keybuf + i*sizeof(example_id));
        rocksdb::Slice key(keybuf + i*sizeof(example_id), sizeof(example_id));
        keys.push_back(key);
    }

    auto status = read_db(db, opt, keys, values);
    free(keybuf);
    return status;
}
}
#endif
