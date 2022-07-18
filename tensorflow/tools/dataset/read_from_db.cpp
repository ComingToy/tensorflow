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

#include "comm_feats_generated.h"
#include "example_generated.h"
#include "feature_generated.h"
#include "absl/strings/str_split.h"
#include "absl/strings/numbers.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"

static rocksdb::Status
open_db(const char* path, rocksdb::DB** db)
{
    rocksdb::Options opt;
    opt.create_if_missing = true;
    opt.max_open_files = 3000;
    opt.write_buffer_size = 500 * 1024 * 1024;
    opt.max_write_buffer_number = 3;
    opt.target_file_size_base = 67108864;

    rocksdb::BlockBasedTableOptions table_opt;
    table_opt.block_cache = rocksdb::NewLRUCache(1000 * (1024 * 1024));
    table_opt.block_cache_compressed = rocksdb::NewLRUCache(500 * (1024 * 1024));
    opt.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_opt));
    return rocksdb::DB::Open(opt, path, db);
}

static void
print_example(dataset::Example const* example)
{
    fprintf(stderr,
            "Get example: example_id = %u, y = %d, z = %d, comm_feat_idx = %s, feat_num = %u, nfeats "
            "= %u\n",
            example->example_id(),
            example->y(),
            example->z(),
            example->comm_feat_id()->c_str(),
            example->feat_num(),
            example->feats()->Length());

    for (auto const& feat : *example->feats()) {
        fprintf(stderr,
                "Get feature: feat_field_id = %u, feat_id = %u, value = %f\n",
                feat->feat_field_id(),
                feat->feat_id(),
                feat->value());
    }
}

static void
print_comm_feats(dataset::CommFeature const* comm_feats)
{
    fprintf(stderr,
            "Get example: comm_feat_idx = %s, feat_num = %u, nfeats "
            "= %u\n",
            comm_feats->comm_feat_id()->c_str(),
            comm_feats->feat_num(),
            comm_feats->feats()->Length());

    for (auto const& feat : *comm_feats->feats()) {
        fprintf(stderr,
                "Get feature: feat_field_id = %u, feat_id = %u, value = %f\n",
                feat->feat_field_id(),
                feat->feat_id(),
                feat->value());
    }
}

ABSL_FLAG(std::string, type, "", "[example|comm_feat]");
ABSL_FLAG(std::string, db, "", "Path to db");
ABSL_FLAG(std::vector<std::string>, keys, {}, "key1,key2...,keyn");
ABSL_FLAG(int64_t, start, -1, "seek to start");
ABSL_FLAG(bool, iter, false, "iterator");
ABSL_FLAG(int64_t, n, 10, "iterator n");

int
main(int argc, char* argv[])
{
    absl::ParseCommandLine(argc, argv);
    auto const& type = absl::GetFlag(FLAGS_type);
    auto const& path_to_db = absl::GetFlag(FLAGS_db);

    if (type.empty() || path_to_db.empty()) {
        fprintf(stderr, "type, db, keys are required\n");
        return -1;
    }

    rocksdb::DB* p;
    auto status = open_db(path_to_db.c_str(), &p);
    if (!status.ok()) {
        fprintf(stderr, "open db %s failed. what: %s\n", argv[1], status.ToString().c_str());
        return -1;
    }
    auto db = std::shared_ptr<rocksdb::DB>(p);
    auto const isexample = type == "example";

    if (absl::GetFlag(FLAGS_iter)) {
        rocksdb::Iterator* it = db->NewIterator(rocksdb::ReadOptions());
        auto const start = absl::GetFlag(FLAGS_start);
        if (start < 0) {
            it->SeekToFirst();
        } else {
            uint32_t u32start = static_cast<uint32_t>(start);
            rocksdb::Slice key = rocksdb::Slice(reinterpret_cast<const char*>(&u32start), sizeof(u32start));
            it->Seek(key);
        }
        fprintf(stderr, "iterate from key %s\n", it->key().ToString(true).c_str());
        int n = 0;
        for (; it->Valid() && n < 10; it->Next(), ++n) {
            fprintf(stderr, "iter key = %s\n", it->key().ToString(true).c_str());
            if (isexample) {
                auto example = dataset::GetExample(it->value().data());
                print_example(example);
            } else {
                auto comm = dataset::GetCommFeature(it->value().data());
                print_comm_feats(comm);
            }
        }

        if (!it->status().ok()) { fprintf(stderr, "iterator error: %s\n", it->status().ToString().c_str()); }

        delete it;

        return 0;
    }

    auto const& keys = absl::GetFlag(FLAGS_keys);

    for (auto const& key : keys) {
        std::string value;
        uint32_t u32key = 0;
        rocksdb::Slice skey;
        if (isexample) {
            if(!absl::SimpleAtoi(key, &u32key)){
                fprintf(stderr, "cannot convert example's key to uint32_t type: key = %s\n", key.data());
                return -1;
            }
            skey = rocksdb::Slice(reinterpret_cast<char*>(&u32key), sizeof(u32key));
        } else {
            skey = rocksdb::Slice(key.data(), key.size());
        }

        fprintf(stderr, "read key: 0x%s\n", skey.ToString(isexample).c_str());
        auto status = db->Get(rocksdb::ReadOptions(), skey, &value);
        if (!status.ok()) {
            fprintf(stderr,
                    "read from %s key = %s failed. message: %s\n",
                    key.data(),
                    skey.ToString(isexample).c_str(),
                    status.ToString().c_str());
            continue;
        }

        if (isexample) {
            auto example = dataset::GetExample(value.data());
            print_example(example);
        } else {
            auto comm_feat = dataset::GetCommFeature(value.data());
            print_comm_feats(comm_feat);
        }
    }
}
