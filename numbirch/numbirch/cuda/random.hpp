/**
 * @file
 */
#pragma once

#include "numbirch/random.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/curand.hpp"
#include "numbirch/common/stl.hpp"
#include "numbirch/macro.hpp"

namespace numbirch {

template<class R>
struct simulate_bernoulli_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE R operator()(const T ρ) const {
    #ifndef __CUDA_ARCH__
    return std::bernoulli_distribution(ρ)(stl<bool>::rng());
    #else
    return false;
    #endif
  }
};

template<class R>
struct simulate_beta_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T α, const U β) const {
    #ifndef __CUDA_ARCH__
    auto u = std::gamma_distribution<R>(α, 1.0)(stl<R>::rng());
    auto v = std::gamma_distribution<R>(β, 1.0)(stl<R>::rng());
    return u/(u + v);
    #else
    return 0.5;
    #endif
  }
};

template<class R>
struct simulate_binomial_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T n, const U ρ) const {
    #ifndef __CUDA_ARCH__
    return std::binomial_distribution<int>(n, ρ)(stl<R>::rng());
    // ^ binomial_distribution requires integral type
    #else
    return 0;
    #endif
  }
};

template<class R>
struct simulate_chi_squared_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE R operator()(const T ν) const {
    #ifndef __CUDA_ARCH__
    return std::chi_squared_distribution<R>(ν)(stl<R>::rng());
    #else
    return 1.0;
    #endif
  }
};

template<class R>
struct simulate_exponential_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE R operator()(const T λ) const {
    #ifndef __CUDA_ARCH__
    return std::exponential_distribution<R>(λ)(stl<R>::rng());
    #else
    return 1.0;
    #endif
  }
};

template<class R>
struct simulate_gamma_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T k, const U θ) const {
    #ifndef __CUDA_ARCH__
    return std::gamma_distribution<long double>(k, θ)(stl<R>::rng());
    #else
    return 1.0;
    #endif
  }
};

template<class R>
struct simulate_gaussian_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T μ, const U σ2) const {
    #ifndef __CUDA_ARCH__
    return std::normal_distribution<R>(μ, std::sqrt(σ2))(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return μ + std::sqrt(σ2)*curand_normal_double(curand_rng());
    } else {
      return μ + std::sqrt(σ2)*curand_normal(curand_rng());
    }
    #endif
  }
};

template<class R>
struct simulate_negative_binomial_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T k, const U ρ) const {
    #ifndef __CUDA_ARCH__
    return std::negative_binomial_distribution<int>(k, ρ)(stl<R>::rng());
    // ^ negative_binomial_distribution requires integral type
    #else
    return 0;
    #endif
  }
};

template<class R>
struct simulate_poisson_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE R operator()(const T λ) const {
    #ifndef __CUDA_ARCH__
    return std::poisson_distribution<int>(λ)(stl<R>::rng());
    // ^ poisson_distribution requires integral type
    #else
    return 0;
    #endif
  }
};

template<class R>
struct simulate_student_t_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE R operator()(const T ν) const {
    #ifndef __CUDA_ARCH__
    return std::student_t_distribution<R>(ν)(stl<R>::rng());
    #else
    return 0.0;
    #endif
  }
};

template<class R>
struct simulate_uniform_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T l, const U u) const {
    #ifndef __CUDA_ARCH__
    return std::uniform_real_distribution<R>(l, u)(stl<R>::rng());
    #else
    return 0.0;
    #endif
  }
};

template<class R>
struct simulate_uniform_int_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T l, const U u) const {
    #ifndef __CUDA_ARCH__
    return std::uniform_int_distribution<int>(l, u)(stl<R>::rng());
    // ^ uniform_int_distribution requires integral type
    #else
    return 0;
    #endif
  }
};

template<class R>
struct simulate_weibull_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T k, const U λ) const {
    #ifndef __CUDA_ARCH__
    return std::weibull_distribution<R>(k, λ)(stl<R>::rng());
    #else
    return 1.0;
    #endif
  }
};

}
