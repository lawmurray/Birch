/**
*@file
 */
#pragma once

#include "numbirch/random.hpp"
#include "numbirch/macro.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/curand.hpp"
#include "numbirch/common/stl.hpp"
#include "numbirch/common/get.hpp"

namespace numbirch {

template<class R>
struct simulate_bernoulli_functor {
  curandState_t* rngs;
  simulate_bernoulli_functor() : rngs(numbirch::rngs) {
    //
  }
  template<class T>
  NUMBIRCH_HOST_DEVICE R operator()(const T rho) {
    #ifndef __CUDA_ARCH__
    return std::bernoulli_distribution(rho)(stl<bool>::rng());
    #else
    if constexpr (std::is_same_v<T,double>) {
      return curand_uniform_double(curand_rng(rngs)) <= rho;
    } else {
      return curand_uniform(curand_rng(rngs)) <= rho;
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
  NUMBIRCH_HOST_DEVICE R operator()(const R alpha, const R beta) {
    R u, v;
    #ifndef __CUDA_ARCH__
    u = std::gamma_distribution<R>(alpha, 1.0)(stl<R>::rng());
    v = std::gamma_distribution<R>(beta, 1.0)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      u = curand_gamma_double(curand_rng(rngs), alpha);
      v = curand_gamma_double(curand_rng(rngs), beta);
    } else {
      u = curand_gamma(curand_rng(rngs), alpha);
      v = curand_gamma(curand_rng(rngs), beta);
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
  NUMBIRCH_HOST_DEVICE R operator()(const T n, const U rho) {
    #ifndef __CUDA_ARCH__
    return std::binomial_distribution<int>(n, rho)(stl<R>::rng());
    // ^ binomial_distribution requires integral type
    #else
    if constexpr (std::is_same_v<U,double>) {
      return curand_binomial_double(curand_rng(rngs), n, rho);
    } else {
      return curand_binomial(curand_rng(rngs), n, rho);
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
  NUMBIRCH_HOST_DEVICE R operator()(const R nu) {
    #ifndef __CUDA_ARCH__
    return std::chi_squared_distribution<R>(nu)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return R(2.0)*curand_gamma_double(curand_rng(rngs), R(0.5)*nu);
    } else {
      return R(2.0)*curand_gamma(curand_rng(rngs), R(0.5)*nu);
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
  NUMBIRCH_HOST_DEVICE R operator()(const R lambda) {
    #ifndef __CUDA_ARCH__
    return std::exponential_distribution<R>(lambda)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return -std::log(curand_uniform_double(curand_rng(rngs)))/lambda;
    } else {
      return -std::log(curand_uniform(curand_rng(rngs)))/lambda;
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
  NUMBIRCH_HOST_DEVICE R operator()(const R k, const R theta) {
    #ifndef __CUDA_ARCH__
    return std::gamma_distribution<R>(k, theta)(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return curand_gamma(curand_rng(rngs), k)*theta;
    } else {
      return curand_gamma_double(curand_rng(rngs), k)*theta;
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
  NUMBIRCH_HOST_DEVICE R operator()(const R mu, const R sigma2) {
    #ifndef __CUDA_ARCH__
    return std::normal_distribution<R>(mu, std::sqrt(sigma2))(stl<R>::rng());
    #else
    if constexpr (std::is_same_v<R,double>) {
      return mu + std::sqrt(sigma2)*curand_normal_double(curand_rng(rngs));
    } else {
      return mu + std::sqrt(sigma2)*curand_normal(curand_rng(rngs));
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
  NUMBIRCH_HOST_DEVICE R operator()(const T k, const U rho) {
    #ifndef __CUDA_ARCH__
    return std::negative_binomial_distribution<int>(k, rho)(stl<R>::rng());
    // ^ negative_binomial_distribution requires integral type
    #else
    U theta, lambda;
    theta = (1 - rho)/rho;
    if constexpr (std::is_same_v<U,double>) {
      lambda = curand_gamma_double(curand_rng(rngs), k);
    } else {
      lambda = curand_gamma(curand_rng(rngs), k);
    }
    return curand_poisson(curand_rng(rngs), lambda*theta);
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
  NUMBIRCH_HOST_DEVICE R operator()(const T lambda) {
    #ifndef __CUDA_ARCH__
    return std::poisson_distribution<int>(lambda)(stl<R>::rng());
    // ^ poisson_distribution requires integral type
    #else
    return curand_poisson(curand_rng(rngs), lambda);
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
  NUMBIRCH_HOST_DEVICE R operator()(const R k, const R lambda) {
    #ifndef __CUDA_ARCH__
    return std::weibull_distribution<R>(k, lambda)(stl<R>::rng());
    #else
    R u;
    if constexpr (std::is_same_v<R,double>) {
      u = curand_uniform_double(curand_rng(rngs));
    } else {
      u = curand_uniform(curand_rng(rngs));
    }
    return lambda*std::pow(-std::log(R(1.0) - u), R(1.0)/k);
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
      R nu = get(k) + n - 1 - i; // i is 0-based here
      R x;
      #ifndef __CUDA_ARCH__
      x = std::chi_squared_distribution<R>(nu)(stl<R>::rng());
      #else
      if constexpr (std::is_same_v<R,double>) {
        x = R(2.0)*curand_gamma_double(curand_rng(rngs), R(0.5)*nu);
      } else {
        x = R(2.0)*curand_gamma(curand_rng(rngs), R(0.5)*nu);
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
