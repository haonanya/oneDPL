//===-- xpu_unique_pred.pass.cpp --------------------------------------------===//
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

struct foo
{
    template <class T>
    bool
    operator()(const T& x, const T& y)
    {
        return x == y;
    }
};

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    int ib[] = {0, 1, 1};
    int ic[] = {0, 1, 1, 1, 2, 2, 2};
    bool ret = true;
    sycl::range<1> item1{1};
    sycl::range<1> item2{3};
    sycl::range<1> item3{7};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        sycl::buffer<int, 1> buffer2(ib, item2);
        sycl::buffer<int, 1> buffer3(ic, item3);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            auto acc_arr1 = buffer2.get_access<sycl_write>(cgh);
            auto acc_arr2 = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<Iter>([=]() {
                Iter r1 = std::unique(Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + 3), foo());
                ret_acc[0] &= (base(r1) == &acc_arr1[0] + 2);
                Iter r2 = std::unique(Iter(&acc_arr2[0]), Iter(&acc_arr2[0] + 7), foo());
                ret_acc[0] &= (base(r2) == &acc_arr2[0] + 3);
            });
        });
    }
    assert(ret);
    assert(ib[0] == 0);
    assert(ib[1] == 1);
    assert(ic[0] == 0);
    assert(ic[1] == 1);
    assert(ic[2] == 2);
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
