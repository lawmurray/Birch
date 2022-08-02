
/**
 * @file
 * 
 * SYCL boilerplate.
 */
#pragma once

#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

namespace numbirch {

extern thread_local sycl::device device;
extern thread_local sycl::context context;
extern thread_local sycl::queue queue;

}
