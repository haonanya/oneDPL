//===-- xpu_reverse.pass.cpp --------------------------------------------===//
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

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    const unsigned N = 5;
    int ia[N] = {0, 1, 2, 3, 4};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<int, 1> buffer1(ia, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ia_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<Iter>([=]() { std::reverse(Iter(&ia_acc[0]), Iter(&ia_acc[0] + N)); });
        });
    }
    // check data

    assert(ia[0] == 4);
    assert(ia[1] == 3);
    assert(ia[2] == 2);
    assert(ia[3] == 1);
    assert(ia[4] == 0);
}

int
main(int, char**)
{
    sycl::queue deviceQueue;
    test<bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<int*>>(deviceQueue);
    test<int*>(deviceQueue);
    return 0;
}
