/**
 * @file
 * 
 * oneMKL integration.
 */
#pragma once

#include <oneapi/mkl.hpp>

namespace mkl = oneapi::mkl;
namespace blas = oneapi::mkl::blas::column_major;
namespace lapack = oneapi::mkl::lapack;
