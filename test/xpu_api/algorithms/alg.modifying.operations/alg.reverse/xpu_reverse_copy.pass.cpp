//===-- xpu_reverse_copy.pass.cpp --------------------------------------------===//
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

template <class Iter1, class Iter2>
class KernelTest;

template <class Iter1, class Iter2>
void
test(sycl::queue& deviceQueue)
{
    const unsigned N = 5;
    int ia[N] = {0, 1, 2, 3, 4};
    int ib[N] = {-1};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<int, 1> buffer1(ia, itemN);
        sycl::buffer<int, 1> buffer2(ib, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ia_acc = buffer1.get_access<sycl_write>(cgh);
            auto ib_acc = buffer2.get_access<sycl_write>(cgh);
            cgh.single_task<KernelTest<Iter1, Iter2>>(
                [=]() { std::reverse_copy(Iter1(&ia_acc[0]), Iter1(&ia_acc[0] + N), Iter2(&ib_acc[0])); });
        });
    }
    // check data

    assert(ib[0] == 4);
    assert(ib[1] == 3);
    assert(ib[2] == 2);
    assert(ib[3] == 1);
    assert(ib[4] == 0);
}

int
main(int, char**)
{
    sycl::queue deviceQueue;
    test<bidirectional_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, int*>(deviceQueue);

    test<random_access_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, int*>(deviceQueue);

    test<const int*, output_iterator<int*>>(deviceQueue);
    test<const int*, forward_iterator<int*>>(deviceQueue);
    test<const int*, bidirectional_iterator<int*>>(deviceQueue);
    test<const int*, random_access_iterator<int*>>(deviceQueue);
    test<const int*, int*>(deviceQueue);

    return 0;
}
