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

#ifndef __ALICCP_DATASET_PARSER_H__
#define __ALICCP_DATASET_PARSER_H__
#include "dataset_parsers.h"
#include "absl/strings/str_split.h"

namespace dataset {
class AliccpDatasetParser : public DatasetParser
{
  public:
    int parse_skeleton_line(flatbuffers::FlatBufferBuilder& builder,
                            std::string const& line,
                            std::vector<char>& key) override
    {
        auto segs = absl::StrSplit(line, ",");
        std::vector<absl::string_view> items(segs);
        if (items.size() != 6) {
            fprintf(stderr, "items.size(=%zu) != 6\n", items.size());
            return -1;
        }

        auto const example_id = static_cast<uint64_t>(std::stoul(items[0].data()));
        auto const y = static_cast<uint16_t>(std::stoi(items[1].data()));
        auto const z = static_cast<uint16_t>(std::stoi(items[2].data()));
        auto const& feat_idx = items[3];
        auto const feat_num = static_cast<uint16_t>(std::stoul(items[4].data()));
        auto const& feats = items[5];
        const char* p = reinterpret_cast<const char*>(&example_id);
        for (auto i = 0; i < sizeof(example_id); ++i) {
            key.push_back(p[i]);
        }

        std::vector<flatbuffers::Offset<dataset::Feature>> vfeats;
        if (parse_feats(builder, feats, vfeats) != 0) { return -1; }

        auto example =
            dataset::CreateExampleDirect(builder, example_id, y, z, feat_idx.data(), feat_num, &vfeats);
        builder.Finish(example);
        return 0;

        return 0;
    }

    int parse_common_line(flatbuffers::FlatBufferBuilder& builder,
                          std::string const& line,
                          std::vector<char>& key) override
    {
        auto segs = absl::StrSplit(line, ",");
        std::vector<absl::string_view> items(segs);

        if (items.size() != 3) {
            fprintf(stderr,
                    "parse_common_line failed. items.size(=%zu) != 3, line=%s\n",
                    items.size(),
                    line.c_str());
            return -1;
        }

        auto const& comm_feat_id = items[0];
        auto const feat_num = static_cast<uint16_t>(std::stoul(items[1].data()));
        auto const& feats = items[2];

        std::copy(comm_feat_id.cbegin(), comm_feat_id.cend(), std::back_inserter(key));
        std::vector<flatbuffers::Offset<dataset::Feature>> vfeats;
        if (parse_feats(builder, feats, vfeats) != 0) {
            fprintf(stderr, "parse comm_feat feats failed. line = %s\n", feats.data());
            return -1;
        }

        auto comm_feats = dataset::CreateCommFeatureDirect(builder, comm_feat_id.data(), feat_num, &vfeats);
        builder.Finish(comm_feats);
        return 0;
    }

  private:
    int parse_feats(flatbuffers::FlatBufferBuilder& builder,
                    absl::string_view const& line,
                    std::vector<flatbuffers::Offset<dataset::Feature>>& vfeats)
    {
        auto segs = absl::StrSplit(line, "\x01");
        std::vector<absl::string_view> feats(segs);

        if (feats.empty()) { return -1; }

        for (auto const& feat : feats) {
            auto kvseg = absl::StrSplit(feat, "\x03");
            std::vector<absl::string_view> kv(kvseg);
            if (kv.size() != 2) {
                fprintf(stderr, "kv.size(=%zu) != 2, feat = %s\n", kv.size(), feat.data());
                continue;
            }

            auto idseg = absl::StrSplit(kv[0], "\x02");
            std::vector<absl::string_view> ids(idseg);

            if (ids.size() != 2) {
                fprintf(stderr, "ids.size(=%zu) != 2, kv[0] = %s", ids.size(), kv[0].data());
                continue;
            }

            auto const feat_field_id = field_to_uint32(ids[0].data());
            auto const feat_id = static_cast<uint32_t>(std::stoul(ids[1].data()));
            auto const value = std::stof(kv[1].data());
            vfeats.push_back(dataset::CreateFeature(builder, feat_field_id, feat_id, value));
        }

        return 0;
    }

    uint32_t field_to_uint32(std::string const& field_id)
    {
        if (field_id.find('_') != std::string::npos) {
            std::string buf;
            std::copy_if(field_id.cbegin(), field_id.cend(), std::back_inserter(buf), [](const char c) {
                return c != '_';
            });
            return std::stoul(buf);
        } else {
            return 100 * std::stoul(field_id);
        }
    }
};

REGISTER_DATASET_PARSER(aliccp, AliccpDatasetParser);
};
#endif
