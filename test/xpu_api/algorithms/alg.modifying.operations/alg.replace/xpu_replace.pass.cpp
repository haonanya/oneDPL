//===-- xpu_replace.pass.cpp --------------------------------------------===//
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

template <typename T1>
class KernelName;

template <class InIter>
void
test(sycl::queue& deviceQueue)
{
    const unsigned N = 5;
    int ia[N] = {0, 1, 2, 3, 4};
    int ib[N] = {0, 1, 2, 3, 4};
    sycl::cl_bool ret = true;
    sycl::range<1> itemN{N};
    {
        sycl::buffer<int, 1> buffer1(ia, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ia_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<KernelName<InIter>>(
                [=]() { std::replace(InIter(ia_acc.get_pointer()), InIter(ia_acc.get_pointer() + N), 2, 5); });
        });
    }
    // check data
    std::replace(std::begin(ib), std::end(ib), 2, 5);
    for (unsigned i = 0; i < N; ++i)
        ret &= (ia[i] == ib[i]);
    assert(ret);
}

int
main(int, char**)
{
    sycl::queue deviceQueue;
    test<forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<int*>>(deviceQueue);
    test<int*>(deviceQueue);
    return 0;
}
