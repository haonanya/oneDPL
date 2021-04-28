//===-- xpu_rotate_copy.pass.cpp --------------------------------------------===//
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

template <class InIter, class OutIter>
void
test(sycl::queue& deviceQueue)
{
    bool ret = true;
    int ia[] = {0, 1, 2, 3};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    int ib[sa] = {0};
    cl::sycl::range<1> itemN{sa};
    {
        cl::sycl::buffer<int, 1> buffer1(ia, itemN);
        cl::sycl::buffer<int, 1> buffer2(ib, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto acc_arr1 = buffer1.get_access<sycl_read>(cgh);
            auto acc_arr2 = buffer2.get_access<sycl_write>(cgh);
            cgh.single_task<KernelTest<InIter, OutIter>>([=]() {
                std::rotate_copy(InIter(&acc_arr1[0]), InIter(&acc_arr1[0] + 2), InIter(&acc_arr1[0] + 4),
                                 OutIter(&acc_arr2[0]));
            });
        });
    }
    assert(ib[0] == 2);
    assert(ib[1] == 3);
    assert(ib[2] == 0);
    assert(ib[3] == 1);
}

int
main(int, char**)
{
    cl::sycl::queue deviceQueue;
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
