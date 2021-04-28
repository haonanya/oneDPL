// <algorithm>

// template <class InputIterator, class OutputIterator1,
//           class OutputIterator2, class Predicate>
//     constexpr pair<OutputIterator1, OutputIterator2>     // constexpr after
//     C++17 partition_copy(InputIterator first, InputIterator last,
//                    OutputIterator1 out_true, OutputIterator2 out_false,
//                    Predicate pred);

#include <CL/sycl.hpp>
#include <algorithm>
#include "checkData.h"
#include "test_iterators.h"

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct is_odd {
  bool operator()(const int &i) const { return i & 1; }
};

template <class Iter1, class Iter2, class KernelName> cl::sycl::cl_bool test() {
  cl::sycl::queue deviceQueue;
  cl::sycl::cl_bool ret = false;
  cl::sycl::cl_bool check = false;
  const int ia[] = {1, 2, 3, 4, 6, 8, 5, 7};
  int r1[10] = {0};
  int r2[10] = {0};
  cl::sycl::range<1> item1{1};
  cl::sycl::range<1> item2{8};
  cl::sycl::range<1> item3{10};
  {
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, item1);
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer2(&check, item1);
    cl::sycl::buffer<int, 1> buffer3(ia, item2);
    cl::sycl::buffer<int, 1> buffer4(r1, item3);
    cl::sycl::buffer<int, 1> buffer5(r2, item3);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto ret_acc = buffer1.get_access<sycl_write>(cgh);
      auto check_acc = buffer2.get_access<sycl_write>(cgh);
      auto acc_arr1 = buffer3.get_access<sycl_write>(cgh);
      auto acc_arr2 = buffer4.get_access<sycl_write>(cgh);
      auto acc_arr3 = buffer5.get_access<sycl_write>(cgh);
      cgh.single_task<KernelName>([=]() {
        // check data transfer between host and device
        const int tmp[] = {1, 2, 3, 4, 6, 8, 5, 7};
        check_acc[0] = checkData(tmp, &acc_arr1[0], 8);
        if (check_acc[0]) {
          typedef std::pair<Iter2, Iter2> P;
          P p = std::partition_copy(
              Iter1(&acc_arr1[0]), Iter1(&acc_arr1[0] + 8), Iter2(&acc_arr2[0]),
              Iter2(&acc_arr3[0]), is_odd());
          ret_acc[0] = (acc_arr2[0] == 1);
          ret_acc[0] &= (acc_arr2[1] == 3);
          ret_acc[0] &= (acc_arr2[2] == 5);
          ret_acc[0] &= (acc_arr2[3] == 7);
          ret_acc[0] &= (acc_arr3[0] == 2);
          ret_acc[0] &= (acc_arr3[1] == 4);
          ret_acc[0] &= (acc_arr3[2] == 6);
          ret_acc[0] &= (acc_arr3[3] == 8);
        }
      });
    });
  }
  // check data after executing kernel function
  const int tmp1[] = {1, 3, 5, 7};
  const int tmp2[] = {2, 4, 6, 8};
  const int tmp3[] = {1, 2, 3, 4, 6, 8, 5, 7};
  check &= checkData(tmp1, r1, 4);
  check &= checkData(tmp2, r2, 4);
  check &= checkData(tmp3, ia, 8);
  if (!check)
    return false;
  return ret;
}

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;
class KernelTest6;
class KernelTest7;
class KernelTest8;
class KernelTest9;
class KernelTest10;
class KernelTest11;
class KernelTest12;
class KernelTest13;
class KernelTest14;
class KernelTest15;
class KernelTest16;
class KernelTest17;
class KernelTest18;
class KernelTest19;
class KernelTest20;
class KernelTest21;
class KernelTest22;
class KernelTest23;
class KernelTest24;
class KernelTest25;

int main(int, char **) {
  auto ret =
      test<input_iterator<const int *>, output_iterator<int *>, KernelTest1>();
  ret &=
      test<input_iterator<const int *>, forward_iterator<int *>, KernelTest2>();
  ret &= test<input_iterator<const int *>, bidirectional_iterator<int *>,
              KernelTest3>();
  ret &= test<input_iterator<const int *>, random_access_iterator<int *>,
              KernelTest4>();
  ret &= test<input_iterator<const int *>, int *, KernelTest5>();

  ret &= test<forward_iterator<const int *>, output_iterator<int *>,
              KernelTest6>();
  ret &= test<forward_iterator<const int *>, forward_iterator<int *>,
              KernelTest7>();
  ret &= test<forward_iterator<const int *>, bidirectional_iterator<int *>,
              KernelTest8>();
  ret &= test<forward_iterator<const int *>, random_access_iterator<int *>,
              KernelTest9>();
  ret &= test<forward_iterator<const int *>, int *, KernelTest10>();

  ret &= test<bidirectional_iterator<const int *>, output_iterator<int *>,
              KernelTest11>();
  ret &= test<bidirectional_iterator<const int *>, forward_iterator<int *>,
              KernelTest12>();
  ret &= test<bidirectional_iterator<const int *>,
              bidirectional_iterator<int *>, KernelTest13>();
  ret &= test<bidirectional_iterator<const int *>,
              random_access_iterator<int *>, KernelTest14>();
  ret &= test<bidirectional_iterator<const int *>, int *, KernelTest15>();

  ret &= test<random_access_iterator<const int *>, output_iterator<int *>,
              KernelTest16>();
  ret &= test<random_access_iterator<const int *>, forward_iterator<int *>,
              KernelTest17>();
  ret &= test<random_access_iterator<const int *>,
              bidirectional_iterator<int *>, KernelTest18>();
  ret &= test<random_access_iterator<const int *>,
              random_access_iterator<int *>, KernelTest19>();
  ret &= test<random_access_iterator<const int *>, int *, KernelTest20>();

  ret &= test<const int *, output_iterator<int *>, KernelTest21>();
  ret &= test<const int *, forward_iterator<int *>, KernelTest22>();
  ret &= test<const int *, bidirectional_iterator<int *>, KernelTest23>();
  ret &= test<const int *, random_access_iterator<int *>, KernelTest24>();
  ret &= test<const int *, int *, KernelTest25>();
  if (ret)
    std::cout << "pass" << std::endl;
  else
    std::cout << "fail" << std::endl;
  return 0;
}
