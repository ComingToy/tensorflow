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

#ifndef __DATASET_PARSER_H__
#define __DATASET_PARSER_H__
#include "comm_feats_generated.h"
#include "example_generated.h"
#include "feature_generated.h"
#include "vocab_generated.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <map>

namespace dataset {

class DatasetParser
{
  public:
    virtual int parse_skeleton_line(flatbuffers::FlatBufferBuilder& builder,
                                    std::string const& line,
                                    std::vector<char>& key) = 0;
    virtual int parse_common_line(flatbuffers::FlatBufferBuilder& builder,
                                  std::string const& line,
                                  std::vector<char>& key) = 0;
};

static std::map<std::string, std::shared_ptr<DatasetParser>> __dataset_parsers;

template<typename ParserType>
class DatasetParserRegister
{
  public:
    template<typename... Args>
    DatasetParserRegister(std::string const& name, Args&&... args)
    {
        assert(__dataset_parsers.count(name) <= 0);
        __dataset_parsers[name] = std::make_shared<ParserType>(std::move(args)...);
    }
};

#define REGISTER_DATASET_PARSER(NAME, TYPE, ...)                                                                     \
    static DatasetParserRegister<TYPE> __dataset_parsers_##NAME##_register(#NAME, ##__VA_ARGS__)

std::shared_ptr<DatasetParser>
get_dataset_parser(std::string const& name)
{
    if (__dataset_parsers.count(name) > 0) { return __dataset_parsers[name]; }
    return nullptr;
}
}

#endif
