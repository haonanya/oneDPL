//===-- xpu_swap_ranges.pass.cpp --------------------------------------------===//
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

template <typename T1, typename T2>
class KernelName;

template <class Iter1, class Iter2>
void
test(sycl::queue& deviceQueue)
{
    const unsigned N = 3;
    int ia[N] = {1, 2, 3};
    int ib[N] = {4, 5, 6};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<int, 1> buffer1(ia, itemN);
        sycl::buffer<int, 1> buffer2(ib, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ia_acc = buffer1.get_access<sycl_write>(cgh);
            auto ib_acc = buffer2.get_access<sycl_write>(cgh);
            cgh.single_task<KernelName<Iter1, Iter2>>(
                [=]() { std::swap_ranges(Iter1(&ia_acc[0]), Iter1(&ia_acc[0] + 3), Iter2(&ib_acc[0])); });
        });
    }
    // check data
    assert(ia[0] == 4);
    assert(ia[1] == 5);
    assert(ia[2] == 6);
    assert(ib[0] == 1);
    assert(ib[1] == 2);
    assert(ib[2] == 3);
}

int
main()
{
    sycl::queue deviceQueue;
    test<forward_iterator<int*>, forward_iterator<int*>>(deviceQueue);
    test<forward_iterator<int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<forward_iterator<int*>, random_access_iterator<int*>>(deviceQueue);
    test<forward_iterator<int*>, int*>(deviceQueue);

    test<bidirectional_iterator<int*>, forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<int*>, random_access_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<int*>, int*>(deviceQueue);

    test<random_access_iterator<int*>, forward_iterator<int*>>(deviceQueue);
    test<random_access_iterator<int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<int*>, random_access_iterator<int*>>(deviceQueue);
    test<random_access_iterator<int*>, int*>(deviceQueue);

    test<int*, forward_iterator<int*>>(deviceQueue);
    test<int*, bidirectional_iterator<int*>>(deviceQueue);
    test<int*, random_access_iterator<int*>>(deviceQueue);
    test<int*, int*>(deviceQueue);
    return 0;
}
