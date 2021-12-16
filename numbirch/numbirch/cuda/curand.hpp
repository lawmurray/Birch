/**
 * @file
 * 
 * cuRAND boilerplate.
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"
#include "numbirch/common/stl.hpp"
#include "numbirch/macro.hpp"

#include <curand_kernel.h>
#include <cassert>

/*
 * Call a cuRAND function and assert success.
 */
#define CURAND_CHECK(call) \
    { \
      curandStatus_t err = call; \
      assert(err == CURAND_STATUS_SUCCESS); \
      if (CUDA_SYNC) { \
        cudaError_t err = cudaStreamSynchronize(stream); \
        assert(err == cudaSuccess); \
      } \
    }


namespace numbirch {
/**
 * @internal
 * 
 * Device pseudorandom number generators for each host thread. The XORWOW
 * generator of cuRAND is used. The MTGP32 generator may be preferred for its
 * longer period (XORWOW: 2^190, MTGP32: 2^11214), but is more difficult to
 * use when implementing rejection sampling algorithms (e.g. custom
 * implementation for gamma variates here), as it can only generate variates
 * for all threads simultaneously. Furthermore, on inspection of
 * curand_uniform.h, it is documented as generating double-precision variates
 * with only 32 random bits, which does not seem ideal.
 */
extern thread_local curandState_t* rngs;

/**
 * @internal
 * 
 * Initialize cuRAND integrations. This should be called during init() by the
 * backend.
 */
void curand_init();

/**
 * @internal
 * 
 * Terminate cuRAND integrations. This should be called during term() by the
 * backend.
 */
void curand_term();

/**
 * @internal
 * 
 * Get the pseudorandom number generator for the current device thread.
 */
NUMBIRCH_DEVICE inline curandState_t* curand_rng(curandState_t* rngs) {
  auto x = threadIdx.x + blockIdx.x*blockDim.x;
  auto y = threadIdx.y + blockIdx.y*blockDim.y;
  return &rngs[x + y*gridDim.x*blockDim.x];
}

template<class T>
NUMBIRCH_DEVICE T curand_gamma_generic(curandState_t* state, const T k) {
  /* see Devroye, L. (1986). Non-Uniform Random Variate Generation.
   * Springer-Verlag, New York, online at
   * http://luc.devroye.org/rnbookindex.html. Specific references below. The
   * same algorithm is used for the k > 1 case in libc++ */
  T a = k;

  /* Best's rejection algorithm works only for a > 1, but we can use the
   * property that, for G ~ Gamma(a + 1) and U ~ Uniform(0, 1), G*pow(U, 1/a)
   * ~ Gamma(a); see Devroye 1986, p. 420 */
  T scale = T(1.0);
  if (a <= T(1.0)) {
    T u;
    if constexpr (std::is_same_v<T,double>) {
      u = curand_uniform_double(state);
    } else {
      u = curand_uniform(state);
    }
    scale = std::pow(u, T(1.0)/a);
    a = k + T(1.0);
  }

  /* Best's rejection algorithm; see Devroye 1986, p. 410 */
  T x;
  T b = a - T(1.0);
  T c = T(3.0)*a - T(0.75);
  bool accept = false;
  do {
    T u, v;
    if constexpr (std::is_same_v<T,double>) {
      u = curand_uniform_double(state);
      v = curand_uniform_double(state);
    } else {
      u = curand_uniform(state);
      v = curand_uniform(state);
    }
    T w = u*(T(1.0) - u);
    if (w != T(0.0)) {
      T y = std::sqrt(c/w)*(u - T(0.5));
      x = b + y;
      if (x >= T(0.0)) {
        T z = T(64.0)*w*w*w*v*v;
        accept = (z <= T(1.0) - T(2.0)*y*y/x) ||
            (std::log(z) <= T(2.0)*(b*std::log(x/b) - y));
      }
    }
  } while (!accept);
  return scale*x;
}

NUMBIRCH_DEVICE inline float curand_gamma(curandState_t* state,
    const float k) {
  return curand_gamma_generic<float>(state, k);
}

NUMBIRCH_DEVICE inline double curand_gamma_double(curandState_t* state,
    const double k) {
  return curand_gamma_generic<double>(state, k);
}

template<class T>
NUMBIRCH_DEVICE int curand_binomial_generic(curandState_t* state, const int n,
    const T rho) {
  /* based on the implementation of std::binomial_distribution in libc++, part
   * of the LLVM project, which cites Kemp, C.D. (1986). A modal method for
   * generating binomial variables. Communication in Statistics- Theory and
   * Methods. 15(3), 805-813 */
  if (n == 0 || rho == 0) {
    return 0;
  } else if (rho == 1) {
    return n;
  } else {
    int ru = int((n + 1)*rho);
    T pu = std::exp(std::lgamma(n + T(1)) - std::lgamma(ru + T(1)) -
        std::lgamma(n - ru + T(1)) + ru*std::log(rho) +
        (n - ru)*std::log(T(1) - rho));
    T u;
    if (std::is_same_v<T,double>) {
      u = curand_uniform_double(state) - pu;
    } else {
      u = curand_uniform(state) - pu;
    }
    if (u < 0) {
      return ru;
    } else {
      int rd = ru;
      T pd = pu;
      T r = rho/(1 - rho);
      while (true) {
        bool b = true;
        if (rd >= 1) {
          pd *= rd/(r*(n - rd + 1));
          u -= pd;
          b = false;
          if (u < 0) {
            return rd - 1;
          }
        }
        if (rd != 0) {
          --rd;
        }
        ++ru;
        if (ru <= n) {
          pu *= (n - ru + 1)*r/ru;
          u -= pu;
          b = false;
          if (u < 0) {
            return ru;
          }
        }
        if (b) {
          return 0;
        }
      }
    }
  }
}

NUMBIRCH_DEVICE inline int curand_binomial(curandState_t* state, const int n,
    const float p) {
  return curand_binomial_generic<float>(state, n, p);
}

NUMBIRCH_DEVICE inline int curand_binomial_double(curandState_t* state,
    const int n, const double p) {
  return curand_binomial_generic<double>(state, n, p);
}

}
