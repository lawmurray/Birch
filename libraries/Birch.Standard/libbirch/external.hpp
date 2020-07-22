/**
 * @file
 *
 * Includes all external headers.
 */
#pragma once

#include <algorithm>
#include <numeric>
#include <limits>
#include <utility>
#include <functional>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <iomanip>
#include <initializer_list>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <unistd.h>
#include <getopt.h>
#include <dlfcn.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/unsupported/Eigen/KroneckerProduct>

/* undefine __SSE2__ as workaround for compile errors for device target due
 * to use of SSE2 intrinsics in Boost headers */
#define OLD_SSE2 __SSE2__
#undef __SSE2__
#include <boost/math/distributions.hpp>
#include <boost/math/special_functions.hpp>
#define __SSE2__ OLD_SSE2


#ifdef _OPENMP
#include <omp.h>
#endif
