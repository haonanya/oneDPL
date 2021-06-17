// -*- C++ -*-
//===-- xpu_reduce.pass.cpp
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
#include <oneapi/dpl/type_traits>

#include <iostream>

#include "support/utils.h"
#include "support/test_iterators.h"

#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

using oneapi::dpl::reduce;

template <typename _T1, typename _T2>
void
ASSERT_EQUAL(_T1&& X, _T2&& Y)
{
    if (X != Y)
        std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    int input[6] = {1, 2, 3, 4, 5, 6};
    int output[4] = {};
    sycl::range<1> numOfItems1{6};
    sycl::range<1> numOfItems2{4};

    {
        sycl::buffer<int, 1> buffer1(input, numOfItems1);
        sycl::buffer<int, 1> buffer2(output, numOfItems2);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto in = buffer1.get_access<sycl_read>(cgh);
            auto out = buffer2.get_access<sycl_write>(cgh);
            cgh.single_task<Iter>([=]() {
                out[0] = reduce(Iter(&in[0]), Iter(&in[0]));
                out[1] = reduce(Iter(&in[0]), Iter(&in[0] + 1));
                out[2] = reduce(Iter(&in[0]), Iter(&in[0] + 2));
                out[3] = reduce(Iter(&in[0]), Iter(&in[0] + 6));
            });
        });
    }
    const int ref[4] = {0, 1, 3, 21};
    for (int i = 0; i < 4; ++i)
    {
        ASSERT_EQUAL(ref[i], output[i]);
    }
}

int
main()
{
#if (_MSC_VER >= 1912 && _MSVC_LANG >= 201703L) ||                                                                     \
    (_GLIBCXX_RELEASE >= 9 && __GLIBCXX__ >= 20190503 && __cplusplus >= 201703L)
    sycl::queue deviceQueue;
    test<input_iterator<const int*>>(deviceQueue);
    test<forward_iterator<const int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>>(deviceQueue);
    test<random_access_iterator<const int*>>(deviceQueue);
    test<const int*>(deviceQueue);
    std::cout << "done" << std::endl;
#else
    std::cout << TestUtils::done(0) << ::std::endl;
#endif
    return 0;
}
