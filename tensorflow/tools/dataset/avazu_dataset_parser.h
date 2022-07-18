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

#ifndef __AVAZU_DATASET_PARSER_H__
#define __AVAZU_DATASET_PARSER_H__
#include "csv_dataset_parser.h"

namespace dataset {

static std::vector<FieldInfo> avazu_field_infos = 
{ 
    {"id", 0, INTERGE, CATEGORY, KEY},
    {"click", 1, INTERGE, CATEGORY, LABEL},
    {"hour", 2, INTERGE, CATEGORY, PROCESS},
    {"C1", 3, INTERGE, CATEGORY, PROCESS},
    {"banner_pos", 4, INTERGE, CATEGORY, PROCESS},
    {"site_id", 5, STRING, CATEGORY, PROCESS},
    {"site_domain", 6, STRING, CATEGORY, PROCESS},
    {"site_category", 7, STRING, CATEGORY, PROCESS},
    {"app_id", 8, STRING, CATEGORY, PROCESS},
    {"app_domain", 9, STRING, CATEGORY, PROCESS},
    {"app_category", 10, STRING, CATEGORY, PROCESS},
    {"device_id", 11, STRING, CATEGORY, PROCESS},
    {"device_ip", 12, STRING, CATEGORY, PROCESS},
    {"device_model", 13, STRING, CATEGORY, PROCESS},
    {"device_type", 14, INTERGE, CATEGORY, PROCESS},
    {"device_conn_type", 15, INTERGE, CATEGORY, PROCESS},
    {"C14", 16, INTERGE, CATEGORY, PROCESS},
    {"C15", 17, INTERGE, CATEGORY, PROCESS},
    {"C16", 18, INTERGE, CATEGORY, PROCESS},
    {"C17", 19, INTERGE, CATEGORY, PROCESS},
    {"C18", 20, INTERGE, CATEGORY, PROCESS},
    {"C19", 21, INTERGE, CATEGORY, PROCESS},
    {"C20", 22, INTERGE, CATEGORY, PROCESS},
    {"C21", 23, INTERGE, CATEGORY, PROCESS},
    {"day", 24, INTERGE, CATEGORY, PROCESS},
};

REGISTER_DATASET_PARSER(avazu, CSVDatasetParser, avazu_field_infos);
};
#endif
