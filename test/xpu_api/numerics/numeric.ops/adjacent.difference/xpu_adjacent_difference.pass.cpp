//===-- xpu_adjacent_difference.pass.cpp
//--------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

#include <oneapi/dpl/numeric>
#include <CL/sycl.hpp>
#include <iostream>

#include "support/test_iterators.h"

template <typename _T1, typename _T2>
void
ASSERT_EQUAL(_T1&& X, _T2&& Y)
{
    if (X != Y)
        std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <class InIter, class OutIter>
class KernelTest;

template <class InIter, class OutIter>
void
test(sycl::queue& deviceQueue)
{
    int input[5] = {15, 10, 6, 3, 1};
    int output[5] = {0};
    sycl::range<1> numOfItems1{5};

    {
        sycl::buffer<int, 1> buffer1(input, numOfItems1);
        sycl::buffer<int, 1> buffer2(output, numOfItems1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto in = buffer1.get_access<sycl_read>(cgh);
            auto out = buffer2.get_access<sycl_write>(cgh);
            cgh.single_task<KernelTest<InIter, OutIter>>([=]() {
                OutIter r = oneapi::dpl::adjacent_difference(InIter(&in[0]), InIter(&in[0] + 5), OutIter(&out[0]));
            });
        });
    }
    int ref[5] = {15, -5, -4, -3, -2};
    for (int i = 0; i < 5; ++i)
        ASSERT_EQUAL(ref[i], output[i]);
}

int
main()
{
    sycl::queue deviceQueue;
    test<input_iterator<const int*>, input_iterator<int*>>(deviceQueue);

    test<input_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, int*>(deviceQueue);

    test<forward_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, int*>(deviceQueue);

    test<bidirectional_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, int*>(deviceQueue);

    test<random_access_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, int*>(deviceQueue);

    test<const int*, input_iterator<int*>>(deviceQueue);
    test<const int*, forward_iterator<int*>>(deviceQueue);
    test<const int*, bidirectional_iterator<int*>>(deviceQueue);
    test<const int*, random_access_iterator<int*>>(deviceQueue);
    test<const int*, int*>(deviceQueue);
    return 0;
}
