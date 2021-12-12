/**
*@file
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
    if constexpr (std::is_same_v<T,double>) {
      return curand_uniform_double(curand_rng()) <= ρ;
    } else {
      return curand_uniform(curand_rng()) <= ρ;
    }
    #endif
  }
};

template<class R>
struct simulate_beta_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R α, const R β) const {
    R u, v;
    #ifndef __CUDA_ARCH__
    u = std::gamma_distribution<R>(α, 1.0)(stl<R>::rng());
    v = std::gamma_distribution<R>(β, 1.0)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      u = curand_gamma_double(curand_rng(), α);
      v = curand_gamma_double(curand_rng(), β);
    } else {
      u = curand_gamma(curand_rng(), α);
      v = curand_gamma(curand_rng(), β);
    }
    #endif
    return u/(u + v);
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
    if constexpr (std::is_same_v<U,double>) {
      return curand_binomial_double(curand_rng(), n, ρ);
    } else {
      return curand_binomial(curand_rng(), n, ρ);
    }
    #endif
  }
};

template<class R>
struct simulate_chi_squared_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R ν) const {
    #ifndef __CUDA_ARCH__
    return std::chi_squared_distribution<R>(ν)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return R(2.0)*curand_gamma_double(curand_rng(), R(0.5)*ν);
    } else {
      return R(2.0)*curand_gamma(curand_rng(), R(0.5)*ν);
    }
    #endif
  }
};

template<class R>
struct simulate_exponential_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R λ) const {
    #ifndef __CUDA_ARCH__
    return std::exponential_distribution<R>(λ)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return -std::log(curand_uniform_double(curand_rng()))/λ;
    } else {
      return -std::log(curand_uniform(curand_rng()))/λ;
    }
    #endif
  }
};

template<class R>
struct simulate_gamma_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R k, const R θ) const {
    #ifndef __CUDA_ARCH__
    return std::gamma_distribution<R>(k, θ)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return curand_gamma(curand_rng(), k)*θ;
    } else {
      return curand_gamma_double(curand_rng(), k)*θ;
    }
    #endif
  }
};

template<class R>
struct simulate_gaussian_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R μ, const R σ2) const {
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
    U θ, λ;
    θ = (1 - ρ)/ρ;
    if constexpr (std::is_same_v<U,double>) {
      λ = curand_gamma_double(curand_rng(), k);
    } else {
      λ = curand_gamma(curand_rng(), k);
    }
    return curand_poisson(curand_rng(), λ*θ);
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
    return curand_poisson(curand_rng(), λ);
    #endif
  }
};

template<class R>
struct simulate_student_t_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R ν) const {
    #ifndef __CUDA_ARCH__
    return std::student_t_distribution<R>(ν)(stl<R>::rng());
    #else
    R λ = R(0.5)*ν*curand_gamma(curand_rng(), R(0.5)*ν);
    if constexpr (std::is_same_v<R,double>) {
      return curand_normal_double(curand_rng())/λ;
    } else {
      return curand_normal(curand_rng())/λ;
    }
    #endif
  }
};

template<class R>
struct simulate_uniform_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R l, const R u) const {
    #ifndef __CUDA_ARCH__
    return std::uniform_real_distribution<R>(l, u)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return l + (u - l)*curand_uniform_double(curand_rng());
    } else {
      return l + (u - l)*curand_uniform(curand_rng());
    }
    #endif
  }
};

template<class R>
struct simulate_uniform_int_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R l, const R u) const {
    #ifndef __CUDA_ARCH__
    return std::uniform_int_distribution<int>(l, u)(stl<R>::rng());
    // ^ uniform_int_distribution requires integral type
    #else
    return int(l + (u - l + 1)*curand_uniform(curand_rng()));
    #endif
  }
};

template<class R>
struct simulate_weibull_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R k, const R λ) const {
    #ifndef __CUDA_ARCH__
    return std::weibull_distribution<R>(k, λ)(stl<R>::rng());
    #else
    R u;
    if constexpr (std::is_same_v<R,double>) {
      u = curand_uniform_double(curand_rng());
    } else {
      u = curand_uniform(curand_rng());
    }
    return λ*std::pow(-std::log(R(1.0) - u), R(1.0)/k);
    #endif
  }
};

}
