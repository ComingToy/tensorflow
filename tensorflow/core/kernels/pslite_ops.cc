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

#include "ps/ps.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tensorflow {
/**
 * @brief 用于tensorflow端的ps::KVWorker生命周期管理RAII类，无功能逻辑
 * \see ps::KVWorker
 *
 * @tparam T
 */
template <class T>
class PSWorker {
 public:
  PSWorker(int customer_id = 0, int appid = 0) : _customer_id(customer_id) {
    ps::Start(customer_id);
    if (std::getenv("DMLC_ROLE") && std::getenv("DMLC_PS_ROOT_URI") &&
        std::getenv("DMLC_PS_ROOT_PORT") && std::getenv("HEAPPROFILE") &&
        std::getenv("DMLC_NUM_SERVER") && std::getenv("DMLC_NUM_WORKER")) {
      std::cerr << "init pslite worker!!!!!!" << std::endl;
      _worker = std::make_shared<ps::KVWorker<T>>(customer_id, appid);
    } else {
      std::cerr << "init pslite worker failed!!!!!!" << std::endl;
      _worker = nullptr;
    }
  }

  virtual ~PSWorker() {
    if (_worker) ps::Finalize(_customer_id);
  }

  std::shared_ptr<ps::KVWorker<T>> worker() { return _worker; }

  mutable std::mutex mtx_;

 private:
  std::shared_ptr<ps::KVWorker<T>> _worker;
  int _customer_id;
};

/**
 * @brief 获取KVWorker实例
 *
 * @return KVWorker实例共享指针(std::shared_ptr)
 * \see ps::KVWorker
 * \see std::shared_ptr
 */
static std::shared_ptr<ps::KVWorker<float>> worker() {
  static PSWorker<float> ps(0, 0);
  return ps.worker();
}

/**
 * @brief  将参数push到参数服务器，一般来说只有rank=0节点启动时会将参数push一次
 * @param keys 参数key，计算方式为::tensorflow::hash(var.name)
 * @param data flat后的参数
 * @param lens 参数值flaten后的长度
 */
void ps_push_data(std::vector<uint64_t> const& keys,
                  std::vector<float> const& data, std::vector<int> const& lens,
                  int cmd) {
  try {
    auto w = worker();
    if (!w) {
      LOG(INFO) << "worker is nullptr";
      return;
    }
    w->Wait(w->Push(keys, data, lens, cmd));
  } catch (dmlc::Error const& e) {
    LOG(INFO) << "Push data error. what: " << e.what();
  }
}

/**
 * @brief 从参数服务器获取参数,
 * 一般来说只有非rank=0节点启动时拉取一次参数，更新阶段使用
 * ps::tensorflow::ps_push_pull_data 来更新参数
 *
 * @param keys 参数key，计算方式为::tensorflow::hash(var.name)
 * @param data flat后的参数
 * @param lens 参数值flaten后的长度
 */
void ps_pull_data(std::vector<uint64_t> const& keys, std::vector<float>& data,
                  std::vector<int>& lens, int cmd) {
  auto w = worker();
  try {
    if (!w) {
      LOG(INFO) << "worker is nullptr";
      return;
    }
    w->Wait(w->Pull(keys, &data, &lens, cmd));
  } catch (dmlc::Error const& e) {
    LOG(INFO) << "Pull data error. what: " << e.what();
  }
}

/**
 * @brief 推送梯度到参数服务器，并pull更新后的参数
 *
 * @param keys 参数key
 * @param data flat后的梯度矩阵
 * @param out  flat后的最新参数
 * @param lens 梯度矩阵flat后的长度
 */
void ps_push_pull_data(std::vector<uint64_t> const& keys,
                       std::vector<float> const& data, std::vector<float>& out,
                       std::vector<int>& lens, int cmd) {
  auto w = worker();
  try {
    if (!w) {
      LOG(INFO) << "worker is nullptr";
      return;
    }
    w->Wait(w->PushPull(keys, data, &out, &lens, cmd));
  } catch (dmlc::Error const& e) {
    LOG(INFO) << "Push data error. what: " << e.what();
  }
}

int ps_my_rank(void) {
  volatile auto w = worker();
  (void)w;
  return ps::MyRank();
}
}  // namespace tensorflow

