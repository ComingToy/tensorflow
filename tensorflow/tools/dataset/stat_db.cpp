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

ABSL_FLAG(std::string, db, "", "path to db");

int
main(int argc, char* argv[])
{
    absl::ParseCommandLine(argc, argv);
    auto const& db = absl::GetFlag(FLAGS_db);
    if (db.empty()) {
        fprintf(stderr, "usage %s <path to db>\n", argv[0]);
        return -1;
    }

    rocksdb::DB* p;
    auto status = open_db(db.c_str(), &p);
    if (!status.ok()) {
        fprintf(stderr, "open db %s failed. what: %s\n", db.c_str(), status.ToString().c_str());
        return -1;
    }

    auto pdb = std::shared_ptr<rocksdb::DB>(p);
    auto* it = pdb->NewIterator(rocksdb::ReadOptions());

    uint64_t key_size = 0;
    uint64_t kv_counts = 0;
    uint64_t values_size = 0;

    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        key_size += it->key().size();
        values_size += it->value().size();
        kv_counts += 1;
    }

    delete it;

    fprintf(stderr,
            "stat %s : total key size = %lu, total values size = %lu, kv counts = %lu\n",
            db.c_str(),
            key_size,
            values_size,
            kv_counts);
    return 0;
}
