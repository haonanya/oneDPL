//===-- xpu_binary_transform.pass.cpp --------------------------------------------===//
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
#include <oneapi/dpl/functional>

#include "support/test_iterators.h"

#include <cassert>
#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <typename T1, typename T2, typename T3>
class KernelName;

template <class InIter1, class InIter2, class OutIter>
void
test(sycl::queue& deviceQueue)
{
    const unsigned N = 5;
    int ia[N] = {0, 1, 2, 3, 4};
    int ib[N] = {1, 2, 3, 4, 5};
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
            cgh.single_task<KernelName<InIter1, InIter2, OutIter>>([=]() {
                OutIter r = std::transform(InIter1(&acc_arr2[0]), InIter1(&acc_arr2[0] + N), InIter2(&acc_arr1[0]),
                                           OutIter(&acc_arr2[0]), std::minus<int>());
                ret_acc[0] = (base(r) == &acc_arr2[0] + N);
            });
        });
    }
    // check data
    for (unsigned i = 0; i < N; ++i)
        ret &= (1 == ib[i]);
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue;
    test<input_iterator<const int*>, input_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, input_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, input_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, input_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, input_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, input_iterator<const int*>, int*>(deviceQueue);

    test<input_iterator<const int*>, forward_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, forward_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, forward_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, forward_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, forward_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, forward_iterator<const int*>, int*>(deviceQueue);

    test<input_iterator<const int*>, bidirectional_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, bidirectional_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, bidirectional_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, bidirectional_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, bidirectional_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, bidirectional_iterator<const int*>, int*>(deviceQueue);

    test<input_iterator<const int*>, random_access_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, random_access_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, random_access_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, random_access_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, random_access_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, random_access_iterator<const int*>, int*>(deviceQueue);

    test<input_iterator<const int*>, const int*, output_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, const int*, input_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, const int*, forward_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, const int*, bidirectional_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, const int*, random_access_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, const int*, int*>(deviceQueue);

    test<forward_iterator<const int*>, input_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, input_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, input_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, input_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, input_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, input_iterator<const int*>, int*>(deviceQueue);

    test<forward_iterator<const int*>, forward_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, forward_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, forward_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, forward_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, forward_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, forward_iterator<const int*>, int*>(deviceQueue);

    test<forward_iterator<const int*>, bidirectional_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, bidirectional_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, bidirectional_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, bidirectional_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, bidirectional_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, bidirectional_iterator<const int*>, int*>(deviceQueue);

    test<forward_iterator<const int*>, random_access_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, random_access_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, random_access_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, random_access_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, random_access_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, random_access_iterator<const int*>, int*>(deviceQueue);

    test<forward_iterator<const int*>, const int*, output_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, const int*, input_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, const int*, forward_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, const int*, bidirectional_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, const int*, random_access_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, const int*, int*>(deviceQueue);

    test<bidirectional_iterator<const int*>, input_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, input_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, input_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, input_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, input_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, input_iterator<const int*>, int*>(deviceQueue);

    test<bidirectional_iterator<const int*>, forward_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, forward_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, forward_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, forward_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, forward_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, forward_iterator<const int*>, int*>(deviceQueue);

    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, bidirectional_iterator<int*>>(
        deviceQueue);
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, random_access_iterator<int*>>(
        deviceQueue);
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, int*>(deviceQueue);

    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, bidirectional_iterator<int*>>(
        deviceQueue);
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, random_access_iterator<int*>>(
        deviceQueue);
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, int*>(deviceQueue);

    test<bidirectional_iterator<const int*>, const int*, output_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, const int*, input_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, const int*, forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, const int*, bidirectional_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, const int*, random_access_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, const int*, int*>(deviceQueue);

    test<random_access_iterator<const int*>, input_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, input_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, input_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, input_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, input_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, input_iterator<const int*>, int*>(deviceQueue);

    test<random_access_iterator<const int*>, forward_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, forward_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, forward_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, forward_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, forward_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, forward_iterator<const int*>, int*>(deviceQueue);

    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, bidirectional_iterator<int*>>(
        deviceQueue);
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, random_access_iterator<int*>>(
        deviceQueue);
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, int*>(deviceQueue);

    test<random_access_iterator<const int*>, random_access_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, random_access_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, random_access_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, random_access_iterator<const int*>, bidirectional_iterator<int*>>(
        deviceQueue);
    test<random_access_iterator<const int*>, random_access_iterator<const int*>, random_access_iterator<int*>>(
        deviceQueue);
    test<random_access_iterator<const int*>, random_access_iterator<const int*>, int*>(deviceQueue);

    test<random_access_iterator<const int*>, const int*, output_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, const int*, input_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, const int*, forward_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, const int*, bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, const int*, random_access_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, const int*, int*>(deviceQueue);

    test<const int*, input_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<const int*, input_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<const int*, input_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<const int*, input_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<const int*, input_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<const int*, input_iterator<const int*>, int*>(deviceQueue);

    test<const int*, forward_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<const int*, forward_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<const int*, forward_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<const int*, forward_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<const int*, forward_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<const int*, forward_iterator<const int*>, int*>(deviceQueue);

    test<const int*, bidirectional_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<const int*, bidirectional_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<const int*, bidirectional_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<const int*, bidirectional_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<const int*, bidirectional_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<const int*, bidirectional_iterator<const int*>, int*>(deviceQueue);

    test<const int*, random_access_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<const int*, random_access_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<const int*, random_access_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<const int*, random_access_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<const int*, random_access_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<const int*, random_access_iterator<const int*>, int*>(deviceQueue);

    test<const int*, const int*, output_iterator<int*>>(deviceQueue);
    test<const int*, const int*, input_iterator<int*>>(deviceQueue);
    test<const int*, const int*, forward_iterator<int*>>(deviceQueue);
    test<const int*, const int*, bidirectional_iterator<int*>>(deviceQueue);
    test<const int*, const int*, random_access_iterator<int*>>(deviceQueue);
    test<const int*, const int*, int*>(deviceQueue);

    return 0;
}
