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
NUMBIRCH_DEVICE inline curandState_t* curand_rng() {
  extern __shared__ curandState_t* shared[];
  curandState_t* rngs = shared[0];
  auto x = threadIdx.x + blockIdx.x*blockDim.x;
  auto y = threadIdx.y + blockIdx.y*blockDim.y;
  return &rngs[x + y*gridDim.x*blockDim.x];
}

template<class T>
NUMBIRCH_DEVICE T curand_gamma_generic(curandState_t* state, const T k) {
  /* based on the implementation of std::gamma_distribution in libc++, part of
   * the LLVM project */
  T x;
  if (k == 1) {
    if constexpr (std::is_same_v<T,double>) {
      x = -std::log(curand_uniform_double(state));
    } else {
      x = -std::log(curand_uniform(state));
    }
  } else if (k > 1) {
    T b = k - 1;
    T c = 3*k - T(0.75);
    while (true) {
      T u, v;
      if constexpr (std::is_same_v<T,double>) {
        u = curand_uniform_double(state);
        v = curand_uniform_double(state);
      } else {
        u = curand_uniform(state);
        v = curand_uniform(state);
      }
      T w = u*(1 - u);
      if (w != 0) {
        T y = std::sqrt(c/w)*(u - T(0.5));
        x = b + y;
        if (x >= 0) {
          T z = 64*w*w*w*v*v;
          if (z <= 1 - 2*y*y/x) {
            break;
          }
          if (std::log(z) <= 2*(b*std::log(x/b) - y)) {
            break;
          }
        }
      }
    }
  } else {
    while (true) {
      T u, v;
      if constexpr (std::is_same_v<T,double>) {
        u = curand_uniform_double(state);
        v = curand_uniform_double(state);
      } else {
        u = curand_uniform(state);
        v = curand_uniform(state);
      }
      if (u <= 1 - k) {
        x = std::pow(u, 1/k);
        if (x <= v) {
          break;
        }
      } else {
        T e = -std::log((1 - u)/k);
        x = std::pow(1 - k + k*e, 1/k);
        if (x <= e + v) {
          break;
        }
      }
    }
  }
  return x;
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
    const T ρ) {
  /* based on the implementation of std::binomial_distribution in libc++, part
   * of the LLVM project, which cites Kemp, C.D. (1986). A modal method for
   * generating binomial variables. Communication in Statistics- Theory and
   * Methods. 15(3), 805-813 */
  if (n == 0 || ρ == 0) {
    return 0;
  } else if (ρ == 1) {
    return n;
  } else {
    int ru = int((n + 1)*ρ);
    T pu = std::exp(std::lgamma(n + T(1)) - std::lgamma(ru + T(1)) -
        std::lgamma(n - ru + T(1)) + ru*std::log(ρ) +
        (n - ru)*std::log(T(1) - ρ));
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
      T r = ρ/(1 - ρ);
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
