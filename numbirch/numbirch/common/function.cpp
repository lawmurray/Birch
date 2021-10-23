/**
 * @file
 * 
 * Explicit template instantiations for special functions used from Eigen.
 * This ensures that Eigen dependency does not propagate to downstream code.
 */
#include "numbirch/function.hpp"

/* for recent versions of CUDA, disables warnings about diag_suppress being
 * deprecated in favor of nv_diag_suppress */
#pragma nv_diag_suppress 20236

#if defined(HAVE_UNSUPPORTED_EIGEN_SPECIALFUNCTIONS)
#include <unsupported/Eigen/SpecialFunctions>
#elif defined(HAVE_EIGEN3_UNSUPPORTED_EIGEN_SPECIALFUNCTIONS)
#include <eigen3/unsupported/Eigen/SpecialFunctions>
#endif

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T digamma(const T x) {
  return Eigen::numext::digamma(x);
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T gamma_p(const T a, const T x) {
  return Eigen::numext::igamma(a, x);
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T gamma_q(const T a, const T x) {
  return Eigen::numext::igammac(a, x);
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T ibeta(const T a, const T b, const T x) {
  return Eigen::numext::betainc(a, b, x);
}

template double digamma(double);
template float digamma(float);

template double gamma_p(double, double);
template float gamma_p(float, float);

template double gamma_q(double, double);
template float gamma_q(float, float);

template double ibeta(double, double, double);
template float ibeta(float, float, float);
