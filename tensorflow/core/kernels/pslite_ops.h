/*
 * Copyright (C) 2021  Lee, Conley <743703241@qq.com>
 * Author: Lee, Conley <743703241@qq.com>
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

#ifndef __PS_OPS_H__
#define __PS_OPS_H__
#include <stdint.h>

#include <vector>
namespace tensorflow {
typedef enum { Push = 0, Pull = 1, PushPull = 2 } OpType;

typedef enum { Delta = 0, Overwrite = 1 } CmdType;

extern void ps_push_data(std::vector<uint64_t> const& keys,
                         std::vector<float> const& data,
                         std::vector<int> const& lens, int cmd = 0);
extern void ps_pull_data(std::vector<uint64_t> const& keys,
                         std::vector<float>& data, std::vector<int>& lens,
                         int cmd = 0);
extern void ps_push_pull_data(std::vector<uint64_t> const& keys,
                              std::vector<float> const& data,
                              std::vector<float>& out, std::vector<int>& lens,
                              int cmd = 0);
extern int ps_my_rank(void);
}  // namespace tensorflow

#endif
