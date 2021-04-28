#include <oneapi/dpl/algorithm>

#include <cassert>
#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

class KernelName;

void
test_copy(sycl::queue& deviceQueue)
{
    const unsigned N = 4;
    int ia[N];
    for (unsigned i = 0; i < N; ++i)
        ia[i] = i;
    int ib[N] = {0};
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<int, 1> buffer2(ia, itemN);
        sycl::buffer<int, 1> buffer3(ib, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto acc_arr1 = buffer2.get_access<sycl_read>(cgh);
            auto acc_arr2 = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<KernelName>([=]() {
                auto r = std::copy((const int*)(acc_arr1.get_pointer()), (const int*)(acc_arr1.get_pointer() + N),
                                   (int*)(acc_arr2.get_pointer()));
            });
        });
    }
}

int
main()
{
    sycl::queue deviceQueue;
    test_copy(deviceQueue);

    return 0;
}
