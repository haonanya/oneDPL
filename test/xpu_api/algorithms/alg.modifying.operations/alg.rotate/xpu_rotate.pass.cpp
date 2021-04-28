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

template <class Iter1>
void
test(sycl::queue& deviceQueue)
{
    const unsigned N = 4;
    int ia[N] = {0, 1, 2, 3};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<int, 1> buffer1(ia, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ia_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<Iter1>(
                [=]() { std::rotate(Iter1(&ia_acc[0]), Iter1(&ia_acc[0] + 1), Iter1(&ia_acc[0] + N)); });
        });
    }
    // check data

    assert(ia[0] == 1);
    assert(ia[1] == 2);
    assert(ia[2] == 3);
    assert(ia[3] == 0);
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
