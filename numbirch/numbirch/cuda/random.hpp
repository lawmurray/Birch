/**
*@file
 */
#pragma once

#include "numbirch/random.hpp"
#include "numbirch/macro.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/curand.hpp"
#include "numbirch/common/stl.hpp"
#include "numbirch/common/element.hpp"

namespace numbirch {

template<class R>
struct simulate_bernoulli_functor {
  curandState_t* rngs;
  simulate_bernoulli_functor() : rngs(numbirch::rngs) {
    //
  }
  template<class T>
  NUMBIRCH_HOST_DEVICE R operator()(const T ρ) {
    #ifndef __CUDA_ARCH__
    return std::bernoulli_distribution(ρ)(stl<bool>::rng());
    #else
    if constexpr (std::is_same_v<T,double>) {
      return curand_uniform_double(curand_rng(rngs)) <= ρ;
    } else {
      return curand_uniform(curand_rng(rngs)) <= ρ;
    }
    #endif
  }
};

template<class R>
struct simulate_beta_functor {
  curandState_t* rngs;
  simulate_beta_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R α, const R β) {
    R u, v;
    #ifndef __CUDA_ARCH__
    u = std::gamma_distribution<R>(α, 1.0)(stl<R>::rng());
    v = std::gamma_distribution<R>(β, 1.0)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      u = curand_gamma_double(curand_rng(rngs), α);
      v = curand_gamma_double(curand_rng(rngs), β);
    } else {
      u = curand_gamma(curand_rng(rngs), α);
      v = curand_gamma(curand_rng(rngs), β);
    }
    #endif
    return u/(u + v);
  }
};

template<class R>
struct simulate_binomial_functor {
  curandState_t* rngs;
  simulate_binomial_functor() : rngs(numbirch::rngs) {
    //
  }
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T n, const U ρ) {
    #ifndef __CUDA_ARCH__
    return std::binomial_distribution<int>(n, ρ)(stl<R>::rng());
    // ^ binomial_distribution requires integral type
    #else
    if constexpr (std::is_same_v<U,double>) {
      return curand_binomial_double(curand_rng(rngs), n, ρ);
    } else {
      return curand_binomial(curand_rng(rngs), n, ρ);
    }
    #endif
  }
};

template<class R>
struct simulate_chi_squared_functor {
  curandState_t* rngs;
  simulate_chi_squared_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R ν) {
    #ifndef __CUDA_ARCH__
    return std::chi_squared_distribution<R>(ν)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return R(2.0)*curand_gamma_double(curand_rng(rngs), R(0.5)*ν);
    } else {
      return R(2.0)*curand_gamma(curand_rng(rngs), R(0.5)*ν);
    }
    #endif
  }
};

template<class R>
struct simulate_exponential_functor {
  curandState_t* rngs;
  simulate_exponential_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R λ) {
    #ifndef __CUDA_ARCH__
    return std::exponential_distribution<R>(λ)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return -std::log(curand_uniform_double(curand_rng(rngs)))/λ;
    } else {
      return -std::log(curand_uniform(curand_rng(rngs)))/λ;
    }
    #endif
  }
};

template<class R>
struct simulate_gamma_functor {
  curandState_t* rngs;
  simulate_gamma_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R k, const R θ) {
    #ifndef __CUDA_ARCH__
    return std::gamma_distribution<R>(k, θ)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return curand_gamma(curand_rng(rngs), k)*θ;
    } else {
      return curand_gamma_double(curand_rng(rngs), k)*θ;
    }
    #endif
  }
};

template<class R>
struct simulate_gaussian_functor {
  curandState_t* rngs;
  simulate_gaussian_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R μ, const R σ2) {
    #ifndef __CUDA_ARCH__
    return std::normal_distribution<R>(μ, std::sqrt(σ2))(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return μ + std::sqrt(σ2)*curand_normal_double(curand_rng(rngs));
    } else {
      return μ + std::sqrt(σ2)*curand_normal(curand_rng(rngs));
    }
    #endif
  }
};

template<class R>
struct simulate_negative_binomial_functor {
  curandState_t* rngs;
  simulate_negative_binomial_functor() : rngs(numbirch::rngs) {
    //
  }
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T k, const U ρ) {
    #ifndef __CUDA_ARCH__
    return std::negative_binomial_distribution<int>(k, ρ)(stl<R>::rng());
    // ^ negative_binomial_distribution requires integral type
    #else
    U θ, λ;
    θ = (1 - ρ)/ρ;
    if constexpr (std::is_same_v<U,double>) {
      λ = curand_gamma_double(curand_rng(rngs), k);
    } else {
      λ = curand_gamma(curand_rng(rngs), k);
    }
    return curand_poisson(curand_rng(rngs), λ*θ);
    #endif
  }
};

template<class R>
struct simulate_poisson_functor {
  curandState_t* rngs;
  simulate_poisson_functor() : rngs(numbirch::rngs) {
    //
  }
  template<class T>
  NUMBIRCH_HOST_DEVICE R operator()(const T λ) {
    #ifndef __CUDA_ARCH__
    return std::poisson_distribution<int>(λ)(stl<R>::rng());
    // ^ poisson_distribution requires integral type
    #else
    return curand_poisson(curand_rng(rngs), λ);
    #endif
  }
};

template<class R>
struct simulate_student_t_functor {
  curandState_t* rngs;
  simulate_student_t_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R ν) {
    #ifndef __CUDA_ARCH__
    return std::student_t_distribution<R>(ν)(stl<R>::rng());
    #else
    R u, x, k = R(0.5)*ν;
    if constexpr (std::is_same_v<R,double>) {
      u = curand_gamma_double(curand_rng(rngs), k);
      x = curand_normal_double(curand_rng(rngs));
    } else {
      u = curand_gamma(curand_rng(rngs), k);
      x = curand_normal(curand_rng(rngs));
    }
    return x*std::sqrt(k/u);
    #endif
  }
};

template<class R>
struct simulate_uniform_functor {
  curandState_t* rngs;
  simulate_uniform_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R l, const R u) {
    #ifndef __CUDA_ARCH__
    return std::uniform_real_distribution<R>(l, u)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return l + (u - l)*curand_uniform_double(curand_rng(rngs));
    } else {
      return l + (u - l)*curand_uniform(curand_rng(rngs));
    }
    #endif
  }
};

template<class R>
struct simulate_uniform_int_functor {
  curandState_t* rngs;
  simulate_uniform_int_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R l, const R u) {
    #ifndef __CUDA_ARCH__
    return std::uniform_int_distribution<int>(l, u)(stl<R>::rng());
    // ^ uniform_int_distribution requires integral type
    #else
    return int(l + (u - l + 1)*curand_uniform(curand_rng(rngs)));
    #endif
  }
};

template<class R>
struct simulate_weibull_functor {
  curandState_t* rngs;
  simulate_weibull_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R k, const R λ) {
    #ifndef __CUDA_ARCH__
    return std::weibull_distribution<R>(k, λ)(stl<R>::rng());
    #else
    R u;
    if constexpr (std::is_same_v<R,double>) {
      u = curand_uniform_double(curand_rng(rngs));
    } else {
      u = curand_uniform(curand_rng(rngs));
    }
    return λ*std::pow(-std::log(R(1.0) - u), R(1.0)/k);
    #endif
  }
};

template<class R>
struct standard_gaussian_functor {
  curandState_t* rngs;
  standard_gaussian_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const int i, const int j) {
    #ifndef __CUDA_ARCH__
    return std::normal_distribution<R>()(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return curand_normal_double(curand_rng(rngs));
    } else {
      return curand_normal(curand_rng(rngs));
    }
    #endif
  }
};

template<class R, class T>
struct standard_wishart_functor {
  T k;
  int n;
  curandState_t* rngs;
  standard_wishart_functor(const T& k, const int n) :
      k(k),
      n(n),
      rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const int i, const int j) {
    if (i == j) {
      /* on diagonal */
      R ν = element(k) + n - i;
      R x;
      #ifndef __CUDA_ARCH__
      x = std::chi_squared_distribution<R>(ν)(stl<R>::rng());
      #else
      if constexpr (std::is_same_v<R,double>) {
        x = R(2.0)*curand_gamma_double(curand_rng(rngs), R(0.5)*ν);
      } else {
        x = R(2.0)*curand_gamma(curand_rng(rngs), R(0.5)*ν);
      }
      #endif
      return std::sqrt(x);
    } else if (i > j) {
      /* in lower triangle */
      #ifndef __CUDA_ARCH__
      return std::normal_distribution<R>()(stl<R>::rng());
      #else
      if constexpr (std::is_same_v<R,double>) {
        return curand_normal_double(curand_rng(rngs));
      } else {
        return curand_normal(curand_rng(rngs));
      }
      #endif
    } else {
      /* in upper triangle */
      return R(0.0);
    }
  }
};

}
