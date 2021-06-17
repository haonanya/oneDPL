// -*- C++ -*-
//===-- xpu_transform_reduce_iter_iter_init_bop_uop.pass.cpp
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
#include <oneapi/dpl/utility>

#include <iostream>
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

struct identity
{
    template <class T>
    constexpr decltype(auto)
    operator()(T&& x) const
    {
        return oneapi::dpl::forward<T>(x);
    }
};

struct twice
{
    template <class T>
    constexpr auto
    operator()(const T& x) const
    {
        return 2 * x;
    }
};

using oneapi::dpl::transform_reduce;

template <class Iter1>
void
test(sycl::queue& deviceQueue)
{
    int input[6] = {1, 2, 3, 4, 5, 6};
    int output[8] = {};
    sycl::range<1> numOfItems1{6};
    sycl::range<1> numOfItems2{8};

    {
        sycl::buffer<int, 1> buffer1(input, numOfItems1);
        sycl::buffer<int, 1> buffer2(output, numOfItems2);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto in = buffer1.get_access<sycl_read>(cgh);
            auto out = buffer2.get_access<sycl_write>(cgh);
            cgh.single_task<Iter1>([=]() {
                out[0] = transform_reduce(Iter1(&in[0]), Iter1(&in[0]), 0, oneapi::dpl::plus<>(), identity());
                out[1] = transform_reduce(Iter1(&in[0]), Iter1(&in[0]), 1, oneapi::dpl::multiplies<>(), identity());
                out[2] = transform_reduce(Iter1(&in[0]), Iter1(&in[0] + 1), 0, oneapi::dpl::multiplies<>(), identity());
                out[3] = transform_reduce(Iter1(&in[0]), Iter1(&in[0] + 1), 2, oneapi::dpl::plus<>(), identity());
                out[4] = transform_reduce(Iter1(&in[0]), Iter1(&in[0] + 6), 4, oneapi::dpl::multiplies<>(), identity());
                out[5] = transform_reduce(Iter1(&in[0]), Iter1(&in[0] + 6), 4, oneapi::dpl::plus<>(), identity());
                out[6] = transform_reduce(Iter1(&in[0]), Iter1(&in[0] + 2), 0, oneapi::dpl::plus<>(), twice());
                out[7] = transform_reduce(Iter1(&in[0]), Iter1(&in[0] + 6), 4, oneapi::dpl::plus<>(), twice());
            });
        });
    }
    int ref[8] = {0, 1, 0, 3, 2880, 25, 6, 46};
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
    test<input_iterator<const int*>>(deviceQueue);
    test<forward_iterator<const int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>>(deviceQueue);
    test<random_access_iterator<const int*>>(deviceQueue);
    test<const int*>(deviceQueue);
    test<int*>(deviceQueue);
    std::cout << "done" << std::endl;
#else
    std::cout << TestUtils::done(0) << ::std::endl;
#endif
    return 0;
}
