/**
*@file
 */
#pragma once

#include "numbirch/utility.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/curand.hpp"

namespace numbirch {

struct simulate_bernoulli_functor {
  curandState_t* rngs;
  simulate_bernoulli_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE bool operator()(const real rho) {
    #ifndef __CUDA_ARCH__
    return std::bernoulli_distribution(rho)(stl<bool>::rng());
    #else
    if constexpr (std::is_same_v<real,double>) {
      return curand_uniform_double(curand_rng(rngs)) <= rho;
    } else {
      return curand_uniform(curand_rng(rngs)) <= rho;
    }
    #endif
  }
};

struct simulate_beta_functor {
  curandState_t* rngs;
  simulate_beta_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE real operator()(const real alpha, const real beta) {
    real u, v;
    #ifndef __CUDA_ARCH__
    auto& rng = stl<real>::rng();
    u = std::gamma_distribution<real>(alpha)(rng);
    v = std::gamma_distribution<real>(beta)(rng);
    #else
    if constexpr (std::is_same_v<real,double>) {
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

struct simulate_binomial_functor {
  curandState_t* rngs;
  simulate_binomial_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE int operator()(const real n, const real rho) {
    #ifndef __CUDA_ARCH__
    return std::binomial_distribution<int>(n, rho)(stl<int>::rng());
    #else
    if constexpr (std::is_same_v<real,double>) {
      return curand_binomial_double(curand_rng(rngs), n, rho);
    } else {
      return curand_binomial(curand_rng(rngs), n, rho);
    }
    #endif
  }
};

struct simulate_chi_squared_functor {
  curandState_t* rngs;
  simulate_chi_squared_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE real operator()(const real nu) {
    #ifndef __CUDA_ARCH__
    return std::chi_squared_distribution<real>(nu)(stl<real>::rng());
    #else
    if constexpr (std::is_same_v<real,double>) {
      return real(2.0)*curand_gamma_double(curand_rng(rngs), real(0.5)*nu);
    } else {
      return real(2.0)*curand_gamma(curand_rng(rngs), real(0.5)*nu);
    }
    #endif
  }
};

struct simulate_exponential_functor {
  curandState_t* rngs;
  simulate_exponential_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE real operator()(const real lambda) {
    #ifndef __CUDA_ARCH__
    return std::exponential_distribution<real>(lambda)(stl<real>::rng());
    #else
    if constexpr (std::is_same_v<real,double>) {
      return -std::log(curand_uniform_double(curand_rng(rngs)))/lambda;
    } else {
      return -std::log(curand_uniform(curand_rng(rngs)))/lambda;
    }
    #endif
  }
};

struct simulate_gamma_functor {
  curandState_t* rngs;
  simulate_gamma_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE real operator()(const real k, const real theta) {
    #ifndef __CUDA_ARCH__
    return std::gamma_distribution<real>(k, theta)(stl<real>::rng());
    #else
    if constexpr (std::is_same_v<real,double>) {
      return curand_gamma(curand_rng(rngs), k)*theta;
    } else {
      return curand_gamma_double(curand_rng(rngs), k)*theta;
    }
    #endif
  }
};

struct simulate_gaussian_functor {
  curandState_t* rngs;
  simulate_gaussian_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE real operator()(const real mu, const real sigma2) {
    #ifndef __CUDA_ARCH__
    return std::normal_distribution<real>(mu, std::sqrt(sigma2))(stl<real>::rng());
    #else
    if constexpr (std::is_same_v<real,double>) {
      return mu + std::sqrt(sigma2)*curand_normal_double(curand_rng(rngs));
    } else {
      return mu + std::sqrt(sigma2)*curand_normal(curand_rng(rngs));
    }
    #endif
  }
};

struct simulate_negative_binomial_functor {
  curandState_t* rngs;
  simulate_negative_binomial_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE real operator()(const real k, const real rho) {
    #ifndef __CUDA_ARCH__
    return std::negative_binomial_distribution<int>(k, rho)(stl<real>::rng());
    // ^ negative_binomial_distribution requires integral type
    #else
    real theta, lambda;
    theta = (real(1.0) - rho)/rho;
    if constexpr (std::is_same_v<real,double>) {
      lambda = curand_gamma_double(curand_rng(rngs), k);
    } else {
      lambda = curand_gamma(curand_rng(rngs), k);
    }
    return curand_poisson(curand_rng(rngs), lambda*theta);
    #endif
  }
};

struct simulate_poisson_functor {
  curandState_t* rngs;
  simulate_poisson_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE int operator()(const real lambda) {
    #ifndef __CUDA_ARCH__
    return std::poisson_distribution<int>(lambda)(stl<real>::rng());
    // ^ poisson_distribution requires integral type
    #else
    return curand_poisson(curand_rng(rngs), lambda);
    #endif
  }
};

struct simulate_uniform_functor {
  curandState_t* rngs;
  simulate_uniform_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE real operator()(const real l, const real u) {
    #ifndef __CUDA_ARCH__
    return std::uniform_real_distribution<real>(l, u)(stl<real>::rng());
    #else
    if constexpr (std::is_same_v<real,double>) {
      return l + (u - l)*curand_uniform_double(curand_rng(rngs));
    } else {
      return l + (u - l)*curand_uniform(curand_rng(rngs));
    }
    #endif
  }
};

struct simulate_uniform_int_functor {
  curandState_t* rngs;
  simulate_uniform_int_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE int operator()(const int l, const int u) {
    #ifndef __CUDA_ARCH__
    return std::uniform_int_distribution<int>(l, u)(stl<int>::rng());
    // ^ uniform_int_distribution requires integral type
    #else
    if constexpr (std::is_same_v<real,double>) {
      return l + int(real(u - l + 1)*curand_uniform_double(curand_rng(rngs)));
    } else {
      return l + int(real(u - l + 1)*curand_uniform(curand_rng(rngs)));
    }
    #endif
  }
};

struct simulate_weibull_functor {
  curandState_t* rngs;
  simulate_weibull_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE real operator()(const real k, const real lambda) {
    #ifndef __CUDA_ARCH__
    return std::weibull_distribution<real>(k, lambda)(stl<real>::rng());
    #else
    real u;
    if constexpr (std::is_same_v<real,double>) {
      u = curand_uniform_double(curand_rng(rngs));
    } else {
      u = curand_uniform(curand_rng(rngs));
    }
    return lambda*std::pow(-std::log(real(1.0) - u), real(1.0)/k);
    #endif
  }
};

struct standard_gaussian_functor {
  curandState_t* rngs;
  standard_gaussian_functor() : rngs(numbirch::rngs) {
    //
  }
  NUMBIRCH_HOST_DEVICE real operator()(const int i, const int j) {
    #ifndef __CUDA_ARCH__
    return std::normal_distribution<real>()(stl<real>::rng());
    #else
    if constexpr (std::is_same_v<real,double>) {
      return curand_normal_double(curand_rng(rngs));
    } else {
      return curand_normal(curand_rng(rngs));
    }
    #endif
  }
};

template<class T>
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
  NUMBIRCH_HOST_DEVICE real operator()(const int i, const int j) {
    if (i == j) {
      /* on diagonal */
      real nu = get(k) + (n - 1 - i); // i is 0-based here
      real x;
      #ifndef __CUDA_ARCH__
      x = std::chi_squared_distribution<real>(nu)(stl<real>::rng());
      #else
      if constexpr (std::is_same_v<real,double>) {
        x = real(2.0)*curand_gamma_double(curand_rng(rngs), real(0.5)*nu);
      } else {
        x = real(2.0)*curand_gamma(curand_rng(rngs), real(0.5)*nu);
      }
      #endif
      return std::sqrt(x);
    } else if (i > j) {
      /* in lower triangle */
      #ifndef __CUDA_ARCH__
      return std::normal_distribution<real>()(stl<real>::rng());
      #else
      if constexpr (std::is_same_v<real,double>) {
        return curand_normal_double(curand_rng(rngs));
      } else {
        return curand_normal(curand_rng(rngs));
      }
      #endif
    } else {
      /* in upper triangle */
      return real(0.0);
    }
  }
};

}
