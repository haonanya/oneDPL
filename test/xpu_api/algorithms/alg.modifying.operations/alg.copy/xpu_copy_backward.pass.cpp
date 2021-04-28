//===-- xpu_copy_backward.pass.cpp --------------------------------------------===//
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

template <class InIter, class OutIter>
void
test_copy_backward(sycl::queue& deviceQueue)
{
    const unsigned N = 1000;
    int ia[N];
    for (unsigned i = 0; i < N; ++i)
        ia[i] = i;
    int ib[N] = {0};
    sycl::cl_bool ret = true;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        sycl::buffer<int, 1> buffer2(ia, itemN);
        sycl::buffer<int, 1> buffer3(ib, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            auto acc_arr1 = buffer2.get_access<sycl_read>(cgh);
            auto acc_arr2 = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<KernelName<InIter, OutIter>>([=]() {
                OutIter r =
                    std::copy_backward(InIter(&acc_arr1[0]), InIter(&acc_arr1[0] + N), OutIter(&acc_arr2[0] + N));
                ret_acc[0] = (base(r) == &acc_arr2[0]);
            });
        });
    }
    // check data
    for (unsigned i = 0; i < N; ++i)
        ret &= (ia[i] == ib[i]);
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue;
    test_copy_backward<bidirectional_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test_copy_backward<bidirectional_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test_copy_backward<bidirectional_iterator<const int*>, int*>(deviceQueue);

    test_copy_backward<random_access_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test_copy_backward<random_access_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test_copy_backward<random_access_iterator<const int*>, int*>(deviceQueue);

    test_copy_backward<const int*, bidirectional_iterator<int*>>(deviceQueue);
    test_copy_backward<const int*, random_access_iterator<int*>>(deviceQueue);
    test_copy_backward<const int*, int*>(deviceQueue);

    return 0;
}
