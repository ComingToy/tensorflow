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

#ifndef __CUSTOM_OP_COMM_H__
#define __CUSTOM_OP_COMM_H__

#include <rocksdb/db.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/options.h>
#include <rocksdb/table.h>
#include <type_traits>
#include "absl/strings/string_view.h"

namespace std {
template<>
struct hash<rocksdb::Slice>
{
    size_t operator()(rocksdb::Slice const& s) const noexcept { return hash_op(s.ToString(true)); }

  private:
    std::hash<std::string> hash_op;
};
}

template<typename T>
::rocksdb::Slice to_slice(T const& v)
{
    return ::rocksdb::Slice(reinterpret_cast<const char*>(&v), sizeof(v));
}

template<typename T>
absl::string_view to_string_view(T const& v)
{
    return absl::string_view(reinterpret_cast<const char*>(&v), sizeof(v));
}

inline ::rocksdb::Status
read_db(std::shared_ptr<rocksdb::DB> db,
        rocksdb::ReadOptions const& opt,
        std::vector<rocksdb::Slice> const& keys,
        std::vector<std::string>& values,
        std::string const& zeros = "")
{
    auto status = db->MultiGet(opt, keys, &values);

    int failed = 0;
    for (auto i = 0; i < keys.size(); ++i) {
        auto s = status[i];
        if (s.IsNotFound()) {
            values[i] = zeros;
            continue;
        }

        if (!s.ok()) { return s; }
    }

    return rocksdb::Status::OK();
}

inline rocksdb::Status
open_db(const char* path, rocksdb::DB** db, rocksdb::Options& opt)
{
    opt.create_if_missing = false;
    opt.max_open_files = -1;
    opt.max_write_buffer_number = 3;
    opt.target_file_size_base = 67108864;
    opt.new_table_reader_for_compaction_inputs = true;
    opt.statistics = rocksdb::CreateDBStatistics();
    opt.stats_dump_period_sec = 10;
    opt.compression = rocksdb::kZlibCompression;

    rocksdb::BlockBasedTableOptions table_opt;
    table_opt.block_cache = rocksdb::NewLRUCache((16 * 1024 * 1024));
    table_opt.block_cache_compressed = rocksdb::NewLRUCache(16 * (1024 * 1024));
    table_opt.cache_index_and_filter_blocks = true;
    table_opt.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
    table_opt.index_type = rocksdb::BlockBasedTableOptions::kHashSearch;
    table_opt.block_size = 4 * 1024;
    opt.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_opt));

    // read_examples_opts_.readahead_size = 1024*1024*1024;
    // read_examples_opts_.fill_cache = true;
    // read_comm_feat_opts_.readahead_size = 1024*1024*1024;
    // read_comm_feat_opts_.fill_cache = true;
    return rocksdb::DB::OpenForReadOnly(opt, path, db);
}

static rocksdb::Status
open_db_for_write(const char* path, rocksdb::DB** db, rocksdb::Options& opt)
{
    opt.create_if_missing = true;
    opt.max_open_files = 3000;
    opt.write_buffer_size = 500 * 1024 * 1024;
    opt.max_write_buffer_number = 3;
    opt.target_file_size_base = 67108864;
    opt.compression = rocksdb::kZlibCompression;

    rocksdb::BlockBasedTableOptions table_opt;
    table_opt.block_cache = rocksdb::NewLRUCache(1000 * (1024 * 1024));
    table_opt.block_cache_compressed = rocksdb::NewLRUCache(500 * (1024 * 1024));
    opt.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_opt));
    return rocksdb::DB::Open(opt, path, db);
}
#endif
