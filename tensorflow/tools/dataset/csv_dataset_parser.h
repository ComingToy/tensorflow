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

#ifndef __CSV_DATASET_PARSER_H__
#define __CSV_DATASET_PARSER_H__
#include "absl/strings/str_split.h"
#include "dataset_parsers.h"

namespace dataset {
typedef enum { INTERGE = 0, FLOAT, STRING } DType;

typedef enum { CATEGORY = 0, NUMBERIC } FType;

typedef enum { DROP = 0, PROCESS, LABEL, KEY } Option;

struct FieldInfo {
  std::string name;
  uint32_t id;
  DType dtype;
  FType ftype;
  Option option;
};

class CSVDatasetParser : public DatasetParser {
 public:
  CSVDatasetParser(std::vector<FieldInfo> const& fields,
                   std::string delimiter = ",")
      : fields_(fields), delimiter_(delimiter) {}

  static void keybuf(uint64_t const& key, std::vector<char>& buf) {
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

  int parse_skeleton_line(flatbuffers::FlatBufferBuilder& builder,
                          std::string const& line,
                          std::vector<char>& key) override {
    auto segs = absl::StrSplit(line, delimiter_);
    std::vector<absl::string_view> items(segs.begin(), segs.end());
    if (items.size() != fields_.size()) {
      return -1;
    }

    key.clear();

    std::vector<flatbuffers::Offset<dataset::Feature>> feats;
    std::vector<uint8_t> labels;

    auto err_handler = [&line](std::exception const& e) {
      std::cerr << "parse error line: " << line << ", what: " << e.what()
                << std::endl;
      return -1;
    };

    for (auto i = 0; i < items.size(); ++i) {
      auto const& field = fields_[i];
      if (field.option == DROP) {
        continue;
      } else if (field.option == LABEL) {
        try {
          auto label = std::stoi(items[i].data());
          labels.push_back(static_cast<uint8_t>(label));
        } catch (std::invalid_argument const& e) {
          return err_handler(e);
        } catch (std::out_of_range const& e) {
          return err_handler(e);
        }
        continue;
      } else if (field.option == KEY) {
        try {
          uint64_t u64key = std::stoul(items[i].data());
          keybuf(u64key, key);
#if 0
          std::cerr << "key str: " << items[i] << ", u64 key: " << u64key;
          u64key = 0;
          bufkey(std::string(key.begin(), key.end()), u64key);
          std::cerr << ", restore key: " << u64key << std::endl;
#endif
        } catch (std::invalid_argument const& e) {
          return err_handler(e);
        }
        continue;
      }

      try {
        uint32_t const field_id = field.id;
        uint32_t const feat_id =
            field.ftype == CATEGORY ? to_feat_id(items[i].data(), field) : 0;
        float const feat_value = field.ftype == CATEGORY
                                     ? 1.0
                                     : to_feat_value(items[i].data(), field);
        feats.push_back(
            dataset::CreateFeature(builder, field_id, feat_id, feat_value));
      } catch (std::invalid_argument const& e) {
        return err_handler(e);
      }
    }

    auto example = dataset::CreateExampleDirect(
        builder, example_couters_, labels[0], labels.size() > 1 ? labels[1] : 0,
        "", feats.size(), &feats);
    builder.Finish(example);

    if (key.empty()) {
      keybuf(example_couters_, key);
    }

    ++example_couters_;
    return 0;
  }

  int parse_common_line(flatbuffers::FlatBufferBuilder& builder,
                        std::string const& line,
                        std::vector<char>& key) override {
    return -1;
  }

 private:
  std::vector<FieldInfo> fields_;
  std::string delimiter_;
  uint64_t example_couters_;

  uint32_t to_feat_id(std::string const& feat, FieldInfo const& field) const {
    static std::hash<std::string> hash_fn;
    if (field.dtype == INTERGE) {
      return static_cast<uint32_t>(std::stol(feat) %
                                   std::numeric_limits<uint32_t>::max());
    } else {
      auto h = hash_fn(feat);
      return static_cast<uint32_t>(h % std::numeric_limits<uint32_t>::max());
    }
  }

  float to_feat_value(std::string const& feat, FieldInfo const& field) const {
    assert(field.dtype == INTERGE || field.dtype == FLOAT);
    return std::stof(feat);
  }
};
}  // namespace dataset
#endif
