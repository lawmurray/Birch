/**
 * @file
 */
#include "numbirch/oneapi/sycl.hpp"

namespace numbirch {

thread_local sycl::device device;
thread_local sycl::context context;
thread_local sycl::queue queue;

}
