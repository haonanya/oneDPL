//===-- xpu_unique_copy.pass.cpp --------------------------------------------===//
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

template <class InIter, class OutIter>
class KernelName;

template <class InIter, class OutIter>
void
test(sycl::queue& deviceQueue)
{
    bool ret = true;
    const int ib[] = {0, 1, 1, 1, 2, 2, 2};
    int jb[7] = {-1};
    sycl::range<1> item1{1};
    sycl::range<1> item2{7};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<int, 1> buffer2(ib, item2);
        sycl::buffer<int, 1> buffer3(jb, item2);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            auto acc_arr1 = buffer2.get_access<sycl_read>(cgh);
            auto acc_arr2 = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<KernelName<InIter, OutIter>>([=]() {
                auto r = std::unique_copy(InIter(&acc_arr1[0]), InIter(&acc_arr1[0] + 7), OutIter(&acc_arr2[0]));
                ret_acc[0] = (base(r) == &acc_arr2[0] + 3);
            });
        });
    }
    assert(ret);
    assert(jb[0] == 0);
    assert(jb[1] == 1);
    assert(jb[2] == 2);
}

int
main()
{
    sycl::queue deviceQueue;
    test<input_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, int*>(deviceQueue);

    test<forward_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, forward_iterator<int*>>(deviceQueue);

    test<forward_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);

    test<forward_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, int*>(deviceQueue);

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
