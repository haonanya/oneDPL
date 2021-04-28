//===-- xpu_fill_n.pass.cpp --------------------------------------------===//
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

class A
{
    char a_;

  public:
    A() {}
    explicit A(char a) : a_(a) {}
    operator unsigned char() const { return 'b'; }

    friend bool
    operator==(const A& x, const A& y)
    {
        return x.a_ == y.a_;
    }
};

template <class Iter>
void
test_int(sycl::queue& deviceQueue)
{
    const unsigned N = 4;
    int ia[N] = {0};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<int, 1> buffer1(ia, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto acc_arr1 = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<Iter>([=]() { std::fill_n(Iter(&acc_arr1[0]), N, 1); });
        });
    }
    bool ret = true;
    // check data
    for (unsigned i = 0; i < N; ++i)
        ret &= (ia[i] == 1);
    assert(ret);
}

void
test_struct(sycl::queue& deviceQueue)
{
    const unsigned N = 3;
    A a[3];
    sycl::range<1> itemN{N};
    {
        sycl::buffer<A, 1> buffer1(a, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto acc_arr1 = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelName>([=]() { std::fill_n(&acc_arr1[0], N, A('a')); });
        });
    }
    bool ret = true;
    // check data
    for (unsigned i = 0; i < N; ++i)
        ret &= (a[i] == A('a'));
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue;
    test_int<forward_iterator<int*>>(deviceQueue);
    test_int<bidirectional_iterator<int*>>(deviceQueue);
    test_int<random_access_iterator<int*>>(deviceQueue);
    test_int<int*>(deviceQueue);

    test_struct(deviceQueue);
    return 0;
}
