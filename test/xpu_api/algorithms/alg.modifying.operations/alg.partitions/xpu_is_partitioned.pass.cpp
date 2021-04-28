//===-- xpu_is_partition.pass.cpp -----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/algorithm>

#include "support/test_iterators.h"

#include <cassert>
#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

struct is_odd {
  bool operator()(const int &i) const { return i & 1; }
};

template <class Iter> 
void test() {
  cl::sycl::queue deviceQueue;
  cl::sycl::cl_bool ret = false;
  const unsigned n = 6;
  const int ia[] = {1, 2, 3, 4, 5, 6};
  const int ib[] = {1, 3, 5, 2, 4, 6};
  cl::sycl::range<1> item1{1};
  cl::sycl::range<1> itemN{n};
  {
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, item1);
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer2(&check, item1);
    cl::sycl::buffer<int, 1> buffer3(ia, itemN);
    cl::sycl::buffer<int, 1> buffer4(ib, itemN);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto ret_acc = buffer1.get_access<sycl_write>(cgh);
      auto check_acc = buffer2.get_access<sycl_write>(cgh);
      auto acc_arr1 = buffer3.get_access<sycl_write>(cgh);
      auto acc_arr2 = buffer4.get_access<sycl_write>(cgh);
      cgh.single_task<KernelName>([=]() {
        // check data transfer between host and device
        const int tmp1[] = {1, 2, 3, 4, 5, 6};
        const int tmp2[] = {1, 3, 5, 2, 4, 6};
        check_acc[0] = checkData(tmp1, &acc_arr1[0], n);
        check_acc[0] &= checkData(tmp2, &acc_arr2[0], n);
        if (check_acc[0]) {
          {
            unary_counting_predicate<is_odd, int> pred((is_odd()));
            ret_acc[0] = (!std::is_partitioned(
                Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + n), std::ref(pred)));
            ret_acc[0] &= (static_cast<std::ptrdiff_t>(pred.count()) <=
                           std::distance(&acc_arr1[0], &acc_arr1[0] + n));
          }
          {
            unary_counting_predicate<is_odd, int> pred((is_odd()));
            ret_acc[0] &= (std::is_partitioned(
                Iter(&acc_arr2[0]), Iter(&acc_arr2[0] + n), std::ref(pred)));
            ret_acc[0] &= (static_cast<std::ptrdiff_t>(pred.count()) <=
                           std::distance(&acc_arr2[0], &acc_arr2[0] + n));
          }
        }
      });
    });
  }
}

int main() {
  test<forward_iterator<int *>>();
  test<bidirectional_iterator<int *>>();
  test<random_access_iterator<int *>>();
  test<int *>();
  return 0;
}
