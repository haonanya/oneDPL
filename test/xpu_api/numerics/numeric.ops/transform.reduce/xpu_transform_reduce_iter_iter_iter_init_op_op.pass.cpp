// -*- C++ -*-
//===-- xpu_transform_reduce_iter_iter_iter_init_op_op.pass.cpp
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

#include <oneapi/dpl/functional>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/type_traits>

#include <iostream>
#include "support/utils.h"
#include "support/test_iterators.h"
#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <typename _T1, typename _T2>
void
ASSERT_EQUAL(_T1&& X, _T2&& Y)
{
    if (X != Y)
        std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

template <class Iter1, class Iter2>
class KernelName;

using oneapi::dpl::transform_reduce;

template <class Iter1, class Iter2>
void
test(sycl::queue& deviceQueue)
{
    int input1[6] = {1, 2, 3, 4, 5, 6};
    unsigned int input2[6] = {2, 4, 6, 8, 10, 12};
    int output[6] = {};
    sycl::range<1> numOfItems1{6};
    sycl::range<1> numOfItems2{8};

    {
        sycl::buffer<int, 1> buffer1(input1, numOfItems1);
        sycl::buffer<unsigned int, 1> buffer2(input2, numOfItems1);
        sycl::buffer<int, 1> buffer3(output, numOfItems2);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto in1 = buffer1.get_access<sycl_read>(cgh);
            auto in2 = buffer2.get_access<sycl_read>(cgh);
            auto out = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<KernelName<Iter1, Iter2>>([=]() {
                out[0] = transform_reduce(Iter1(&in1[0]), Iter1(&in1[0]), Iter2(&in2[0]), 0, oneapi::dpl::plus<>(),
                                          oneapi::dpl::multiplies<>());
                out[1] = transform_reduce(Iter2(&in2[0]), Iter2(&in2[0]), Iter1(&in1[0]), 1,
                                          oneapi::dpl::multiplies<>(), oneapi::dpl::plus<>());
                out[2] = transform_reduce(Iter1(&in1[0]), Iter1(&in1[0] + 1), Iter2(&in2[0]), 0,
                                          oneapi::dpl::multiplies<>(), oneapi::dpl::plus<>());
                out[3] = transform_reduce(Iter2(&in2[0]), Iter2(&in2[0] + 1), Iter1(&in1[0]), 2, oneapi::dpl::plus<>(),
                                          oneapi::dpl::multiplies<>());
                out[4] = transform_reduce(Iter1(&in1[0]), Iter1(&in1[0] + 2), Iter2(&in2[0]), 0, oneapi::dpl::plus<>(),
                                          oneapi::dpl::multiplies<>());
                out[5] = transform_reduce(Iter2(&in2[0]), Iter2(&in2[0] + 2), Iter1(&in1[0]), 3,
                                          oneapi::dpl::multiplies<>(), oneapi::dpl::plus<>());
                out[6] = transform_reduce(Iter1(&in1[0]), Iter1(&in1[0] + 6), Iter2(&in2[0]), 4,
                                          oneapi::dpl::multiplies<>(), oneapi::dpl::plus<>());
                out[7] = transform_reduce(Iter2(&in2[0]), Iter2(&in2[0] + 6), Iter1(&in1[0]), 4, oneapi::dpl::plus<>(),
                                          oneapi::dpl::multiplies<>());
            });
        });
    }
    int ref[8] = {0, 1, 0, 4, 10, 54, 2099520, 186};
    for (int i = 0; i < 8; ++i)
    {

        ASSERT_EQUAL(ref[i], output[i]);
    }
}

int
main()
{
#if (_MSC_VER >= 1912 && _MSVC_LANG >= 201703L) ||                                                                     \
    (_GLIBCXX_RELEASE >= 9 && __GLIBCXX__ >= 20190503 && __cplusplus >= 201703L)
    //  All the iterator categories
    sycl::queue deviceQueue;
    test<input_iterator<const int*>, input_iterator<const unsigned int*>>(deviceQueue);
    test<input_iterator<const int*>, forward_iterator<const unsigned int*>>(deviceQueue);
    test<input_iterator<const int*>, bidirectional_iterator<const unsigned int*>>(deviceQueue);
    test<input_iterator<const int*>, random_access_iterator<const unsigned int*>>(deviceQueue);

    test<forward_iterator<const int*>, input_iterator<const unsigned int*>>(deviceQueue);
    test<forward_iterator<const int*>, forward_iterator<const unsigned int*>>(deviceQueue);
    test<forward_iterator<const int*>, bidirectional_iterator<const unsigned int*>>(deviceQueue);
    test<forward_iterator<const int*>, random_access_iterator<const unsigned int*>>(deviceQueue);

    test<bidirectional_iterator<const int*>, input_iterator<const unsigned int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, forward_iterator<const unsigned int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const unsigned int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, random_access_iterator<const unsigned int*>>(deviceQueue);

    test<random_access_iterator<const int*>, input_iterator<const unsigned int*>>(deviceQueue);
    test<random_access_iterator<const int*>, forward_iterator<const unsigned int*>>(deviceQueue);
    test<random_access_iterator<const int*>, bidirectional_iterator<const unsigned int*>>(deviceQueue);
    test<random_access_iterator<const int*>, random_access_iterator<const unsigned int*>>(deviceQueue);

    //  Just plain pointers (const vs. non-const, too)
    test<const int*, const unsigned int*>(deviceQueue);
    test<const int*, unsigned int*>(deviceQueue);
    test<int*, const unsigned int*>(deviceQueue);
    test<int*, unsigned int*>(deviceQueue);
    std::cout << "done" << std::endl;
#else
    std::cout << TestUtils::done(0) << ::std::endl;
#endif
    return 0;
}
