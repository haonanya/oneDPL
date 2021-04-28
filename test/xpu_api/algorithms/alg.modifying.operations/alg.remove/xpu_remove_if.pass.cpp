//===-- xpu_remove_if.pass.cpp --------------------------------------------===//
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

struct equal2
{
    bool
    operator()(int i)
    {
        return i == 2;
    }
};

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    const unsigned N = 9;
    int ia[N] = {0, 1, 2, 3, 4, 2, 3, 4, 2};
    sycl::cl_bool ret = true;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        sycl::buffer<int, 1> buffer2(ia, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            auto acc_arr1 = buffer2.get_access<sycl_write>(cgh);
            cgh.single_task<Iter>([=]() {
                Iter r = std::remove_if(Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + N), equal2());
                ret_acc[0] &= (base(r) == &acc_arr1[0] + N - 3);
            });
        });
    }
    // check data
    ret &= (ia[0] == 0);
    ret &= (ia[1] == 1);
    ret &= (ia[2] == 3);
    ret &= (ia[3] == 4);
    ret &= (ia[4] == 3);
    ret &= (ia[5] == 4);
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue;
    test<forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<int*>>(deviceQueue);
    test<int*>(deviceQueue);

    return 0;
}
