// <algorithm>

// template<BidirectionalIterator Iter, Predicate<auto, Iter::value_type> Pred>
//   requires ShuffleIterator<Iter>
//         && CopyConstructible<Pred>
//   Iter
//   partition(Iter first, Iter last, Pred pred);

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
  int ia[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  cl::sycl::range<1> item1{1};
  cl::sycl::range<1> itemN{sa};
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
        int tmp[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        check_acc[0] = checkData(tmp, &acc_arr1[0], sa);
        if (check_acc[0]) {
          Iter r = std::partition(Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + sa),
                                  is_odd());
          ret_acc[0] = (base(r) == &acc_arr1[0] + 5);
          for (int *i = &acc_arr1[0]; i < base(r); ++i)
            ret_acc[0] &= (is_odd()(*i));
          for (int *i = base(r); i < &acc_arr1[0] + sa; ++i)
            ret_acc[0] &= (!is_odd()(*i));
          // check empty
          r = std::partition(Iter(&acc_arr1[0]), Iter(&acc_arr1[0]), is_odd());
          ret_acc[0] &= (base(r) == &acc_arr1[0]);
          // check all false
          for (unsigned i = 0; i < sa; ++i)
            acc_arr1[i] = 2 * i;
          r = std::partition(Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + sa),
                             is_odd());
          ret_acc[0] &= (base(r) == &acc_arr1[0]);
          // check all true
          for (unsigned i = 0; i < sa; ++i)
            acc_arr1[i] = 2 * i + 1;
          r = std::partition(Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + sa),
                             is_odd());
          ret_acc[0] &= (base(r) == &acc_arr1[0] + sa);
          // check all true but last
          for (unsigned i = 0; i < sa; ++i)
            acc_arr1[i] = 2 * i + 1;
          acc_arr1[sa - 1] = 10;
          r = std::partition(Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + sa),
                             is_odd());
          ret_acc[0] &= (base(r) == &acc_arr1[0] + sa - 1);
          for (int *i = &acc_arr1[0]; i < base(r); ++i)
            ret_acc[0] &= (is_odd()(*i));
          for (int *i = base(r); i < &acc_arr1[0] + sa; ++i)
            ret_acc[0] &= (!is_odd()(*i));
          // check all true but first
          for (unsigned i = 0; i < sa; ++i)
            acc_arr1[i] = 2 * i + 1;
          acc_arr1[0] = 10;
          r = std::partition(Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + sa),
                             is_odd());
          ret_acc[0] &= (base(r) == &acc_arr1[0] + sa - 1);
          for (int *i = &acc_arr1[0]; i < base(r); ++i)
            ret_acc[0] &= (is_odd()(*i));
          for (int *i = base(r); i < &acc_arr1[0] + sa; ++i)
            ret_acc[0] &= (!is_odd()(*i));
          // check all false but last
          for (unsigned i = 0; i < sa; ++i)
            acc_arr1[i] = 2 * i;
          acc_arr1[sa - 1] = 11;
          r = std::partition(Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + sa),
                             is_odd());
          ret_acc[0] &= (base(r) == &acc_arr1[0] + 1);
          for (int *i = &acc_arr1[0]; i < base(r); ++i)
            ret_acc[0] &= (is_odd()(*i));
          for (int *i = base(r); i < &acc_arr1[0] + sa; ++i)
            ret_acc[0] &= (!is_odd()(*i));
          // check all false but first
          for (unsigned i = 0; i < sa; ++i)
            acc_arr1[i] = 2 * i;
          acc_arr1[0] = 11;
          r = std::partition(Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + sa),
                             is_odd());
          ret_acc[0] &= (base(r) == &acc_arr1[0] + 1);
          for (int *i = &acc_arr1[0]; i < base(r); ++i)
            ret_acc[0] &= (is_odd()(*i));
          for (int *i = base(r); i < &acc_arr1[0] + sa; ++i)
            ret_acc[0] &= (!is_odd()(*i));
        }
      });
    });
  }
  // check data after executing kernel function
  int tmp[] = {11, 4, 6, 7, 10, 12, 14, 16, 18};
  check &= checkData(tmp, ia, sa);
  if (!check)
    return false;
  return ret;
}

class KernelTest1;
class KernelTest2;
class KernelTest3;

int main(int, char **) {
  auto ret = test<bidirectional_iterator<int *>, KernelTest1>();
  ret &= test<random_access_iterator<int *>, KernelTest2>();
  ret &= test<int *, KernelTest3>();
  if (ret)
    std::cout << "pass" << std::endl;
  else
    std::cout << "fail" << std::endl;
  return 0;
}
