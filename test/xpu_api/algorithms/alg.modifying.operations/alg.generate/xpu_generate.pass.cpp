//===-- xpu_generate.pass.cpp --------------------------------------------===//
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

struct gen_test
{
    int
    operator()() const
    {
        return 1;
    }
};

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    const unsigned N = 4;
    int ia[N] = {0};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<int, 1> buffer1(ia, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto acc_arr1 = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<Iter>([=]() { std::generate(Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + N), gen_test()); });
        });
    }
    bool ret = true;
    // check data
    for (unsigned i = 0; i < N; ++i)
        ret &= (ia[i] == 1);
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue;
    test<forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<int*>>(deviceQueue);
    test<int*>(deviceQueue);

    return 0;
}
