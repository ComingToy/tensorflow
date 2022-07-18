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

#include "aliccp_dataset_parser.h"
#include "avazu_dataset_parser.h"
#include "comm_feats_generated.h"
#include "example_generated.h"
#include "feature_generated.h"
#include "vocab_generated.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "fstream"
#include "iostream"
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
    opt.compression = rocksdb::kZlibCompression;

    rocksdb::BlockBasedTableOptions table_opt;
    table_opt.block_cache = rocksdb::NewLRUCache(1000 * (1024 * 1024));
    table_opt.block_cache_compressed = rocksdb::NewLRUCache(500 * (1024 * 1024));
    opt.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_opt));
    return rocksdb::DB::Open(opt, path, db);
}

ABSL_FLAG(std::string, common_data, "", "path to common feats data");
ABSL_FLAG(std::string, examples_data, "", "path to examples data");
ABSL_FLAG(std::string, common_db, "", "path to common feats db");
ABSL_FLAG(std::string, examples_db, "", "path to examples db");
ABSL_FLAG(int32_t, batch, 10000, "batch size");
ABSL_FLAG(std::string, dataset, "", "name of dataset = {aliccp, avazu}");

static int
write_features_to_db(const std::string& path_to_data,
                     const std::string& path_to_db,
                     const int batch_size,
                     bool const isexample)
{
    rocksdb::DB* db = nullptr;
    auto status = open_db(path_to_db.c_str(), &db);
    if (!status.ok()) {
        fprintf(stderr, "open db failed: %s, msg: %s\n", path_to_db.c_str(), status.ToString().c_str());
        return -1;
    }

    auto pdb = std::shared_ptr<rocksdb::DB>(db);

    rocksdb::WriteOptions option;

    std::ifstream ifs(path_to_data);
    std::string line;
    int cnt = 0;
    flatbuffers::FlatBufferBuilder builder(0);

    auto batch = std::make_shared<rocksdb::WriteBatch>();
    auto start = time(nullptr);
    uint64_t total_size = 0;
    auto parser = dataset::get_dataset_parser(absl::GetFlag(FLAGS_dataset));
    while (std::getline(ifs, line)) {
        std::vector<char> keybuf;
        int err = 0;
        if (isexample) {
            err = parser->parse_skeleton_line(builder, line, keybuf);
        } else {
            err = parser->parse_common_line(builder, line, keybuf);
        }

        if (err < 0) continue;

        auto buf = builder.GetBufferPointer();
	auto buf_size = builder.GetSize();

        rocksdb::Slice value(reinterpret_cast<char*>(buf), buf_size);

        total_size += (keybuf.size() + value.size());
        rocksdb::Slice key(keybuf.data(), keybuf.size());
        batch->Put(key, value);
        ++cnt;
        builder.Clear();

        if (cnt % batch_size == 0) {
            pdb->Write(option, &(*batch));
            fprintf(stderr,
                    "write %s db  batch size = %d, cnt = %d, total_writen_size = %lu, cost %ld seconds\n",
                    path_to_db.c_str(),
                    batch_size,
                    cnt,
                    total_size,
                    time(nullptr) - start);
            start = time(nullptr);
            batch = std::make_shared<rocksdb::WriteBatch>();
        }
    }
    if (batch->GetDataSize() > 0) pdb->Write(option, &(*batch));
    return 0;
}

int
main(int argc, char* argv[])
{
    absl::ParseCommandLine(argc, argv);
    auto const& examples_db = absl::GetFlag(FLAGS_examples_db);
    auto const& examples_data = absl::GetFlag(FLAGS_examples_data);

    if (examples_db.empty() || examples_data.empty()) {
        fprintf(stderr, "type, data, db are required.\n");
        return -1;
    }

    auto const& common_data = absl::GetFlag(FLAGS_common_data);
    auto const& common_db = absl::GetFlag(FLAGS_common_db);

    auto const batch = absl::GetFlag(FLAGS_batch);
    if (!common_data.empty() && !common_db.empty()) {
        write_features_to_db(common_data, common_db, batch, false);
    }

    write_features_to_db(examples_data, examples_db, batch, true);
}
