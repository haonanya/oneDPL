// <algorithm>

// template<class ForwardIterator, class Predicate>
//     constpexr ForwardIterator       // constexpr after C++17
//     partition_point(ForwardIterator first, ForwardIterator last, Predicate
//     pred);

#include <CL/sycl.hpp>
#include <algorithm>
#include "checkData.h"
#include "test_iterators.h"

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct is_odd {
  bool operator()(const int &i) const { return i & 1; }
};

template <class Iter, class KernelName> cl::sycl::cl_bool test() {
  cl::sycl::queue deviceQueue;
  cl::sycl::cl_bool ret = false;
  cl::sycl::cl_bool check = false;
  const unsigned n = 6;
  const int ia[] = {1, 3, 5, 7, 2, 4};
  cl::sycl::range<1> item1{1};
  cl::sycl::range<1> itemN{n};
  {
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, item1);
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer2(&check, item1);
    cl::sycl::buffer<int, 1> buffer3(ia, itemN);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto ret_acc = buffer1.get_access<sycl_write>(cgh);
      auto check_acc = buffer2.get_access<sycl_write>(cgh);
      auto acc_arr1 = buffer3.get_access<sycl_write>(cgh);
      cgh.single_task<KernelName>([=]() {
        // check data transfer between host and device
        const int tmp[] = {1, 3, 5, 7, 2, 4};
        check_acc[0] = checkData(tmp, &acc_arr1[0], n);
        if (check_acc[0]) {
          {
            ret_acc[0] = (std::partition_point(
                              Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + n),
                              is_odd()) == Iter(&acc_arr1[0] + 4));
          }
        }
      });
    });
  }
  // check data after executing kernel function
  const int tmp[] = {1, 3, 5, 7, 2, 4};
  check &= checkData(tmp, ia, n);
  if (!check)
    return false;
  return ret;
}

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;

int main(int, char **) {
  auto ret = test<forward_iterator<int *>, KernelTest1>();
  ret &= test<bidirectional_iterator<int *>, KernelTest2>();
  ret &= test<random_access_iterator<int *>, KernelTest3>();
  ret &= test<int *, KernelTest4>();
  if (ret)
    std::cout << "pass" << std::endl;
  else
    std::cout << "fail" << std::endl;
  return 0;
}
