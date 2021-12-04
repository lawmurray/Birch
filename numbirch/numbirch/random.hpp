/**
 * @file
 * 
 * NumBirch random number generation interface.
 */
#pragma once

#include "numbirch/array/Array.hpp"
#include "numbirch/array/Scalar.hpp"
#include "numbirch/array/Vector.hpp"
#include "numbirch/array/Matrix.hpp"

namespace numbirch {
/**
 * Seed pseudorandom number generators.
 * 
 * @ingroup random
 * 
 * @param s Seed, $s$.
 * 
 * If there are $N$ host threads, then the pseudorandom number generator for
 * the $n$th thread is seeded with $sN + n$. This ensures that different seeds
 * produce disjoint sets of pseudorandom number streams.
 * 
 * According to the backend, there will be multiple pseudorandom number
 * generators---often 32-bit and 64-bit versions on host, and many streams on
 * device. As these use different algorithms, or at least different
 * parameterizations of the same algorithms, they can all use the same seed.
 */
void seed(const int s);

/**
 * Simulate a Bernoulli distribution.
 *
 * @ingroup random
 * 
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param ρ Probability of success.
 * 
 * @return Variate.
 */
template<class R, class T, class = std::enable_if_t<
    is_numeric_v<R> && is_numeric_v<T>,int>>
explicit_t<R,T> simulate_bernoulli(const T& ρ);

/**
 * Simulate a Bernoulli distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * 
 * @param ρ Probability of success.
 * 
 * @return Variate.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
explicit_t<bool,T> simulate_bernoulli(const T& ρ) {
  return simulate_bernoulli<bool,T,int>(ρ);
}

/**
 * Simulate a beta distribution.
 *
 * @ingroup random
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param α First shape.
 * @param β Second shape.
 * 
 * @return Variate.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,implicit_t<T,U>> simulate_beta(const T& α, const U& β);

/**
 * Simulate a beta distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param α First shape.
 * @param β Second shape.
 * 
 * @return Variate.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
default_t<implicit_t<T,U>> simulate_beta(const T& α, const U& β) {
  return simulate_beta<value_t<default_t<implicit_t<T,U>>>,T,U,int>(α, β);
}

/**
 * Simulate a binomial distribution.
 *
 * @ingroup random
 * 
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param n Number of trials.
 * @param ρ Probability of success.
 * 
 * @return Variate.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_numeric_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,implicit_t<T,U>> simulate_binomial(const T& n, const U& ρ);

/**
 * Simulate a binomial distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param n Number of trials.
 * @param ρ Probability of success.
 * 
 * @return Variate.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<int,implicit_t<T,U>> simulate_binomial(const T& n, const U& ρ) {
  return simulate_binomial<int,T,U,int>(n, ρ);
}

/**
 * Simulate a $\chi^2$ distribution.
 *
 * @ingroup random
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param ν Degrees of freedom.
 * 
 * @return Variate.
 */
template<class R, class T, class = std::enable_if_t<
    is_floating_point_v<R> && is_numeric_v<T>,int>>
explicit_t<R,T> simulate_chi_squared(const T& ν);

/**
 * Simulate a $\chi^2$ distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * 
 * @param ν Degrees of freedom.
 * 
 * @return Variate.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> simulate_chi_squared(const T& ν) {
  return simulate_chi_squared<value_t<default_t<T>>,T,int>(ν);
}

/**
 * Simulate an exponential distribution.
 *
 * @ingroup random
 * 
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param λ Rate.
 * 
 * @return Variate.
 */
template<class R, class T, class = std::enable_if_t<
    is_floating_point_v<R> && is_numeric_v<T>,int>>
explicit_t<R,T> simulate_exponential(const T& λ);

/**
 * Simulate an exponential distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * 
 * @param λ Rate.
 * 
 * @return Variate.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> simulate_exponential(const T& λ) {
  return simulate_exponential<value_t<default_t<T>>,T,int>(λ);
}

/**
 * Simulate a gamma distribution.
 *
 * @ingroup random
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param k Shape.
 * @param θ Scale.
 * 
 * @return Variate.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,implicit_t<T,U>> simulate_gamma(const T& k, const U& θ);

/**
 * Simulate a gamma distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param k Shape.
 * @param θ Scale.
 * 
 * @return Variate.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
default_t<implicit_t<T,U>> simulate_gamma(const T& k, const U& θ) {
  return simulate_gamma<value_t<default_t<implicit_t<T,U>>>,T,U,int>(k, θ);
}

/**
 * Simulate a Gaussian distribution.
 *
 * @ingroup random
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param μ Mean.
 * @param σ2 Variance.
 * 
 * @return Variate.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,implicit_t<T,U>> simulate_gaussian(const T& μ, const U& σ2);

/**
 * Simulate a Gaussian distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param μ Mean.
 * @param σ2 Variance.
 * 
 * @return Variate.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
default_t<implicit_t<T,U>> simulate_gaussian(const T& μ, const U& σ2) {
  return simulate_gaussian<value_t<default_t<implicit_t<T,U>>>,T,U,int>(μ,
      σ2);
}

/**
 * Simulate a negative binomial distribution.
 *
 * @ingroup random
 * 
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param k Number of successes.
 * @param ρ Probability of success.
 * 
 * @return Number of failures before `k` number of successes are achieved.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_numeric_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,implicit_t<T,U>> simulate_negative_binomial(const T& k,
    const U& ρ);

/**
 * Simulate a negative binomial distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param k Number of successes.
 * @param ρ Probability of success.
 * 
 * @return Number of failures before `k` number of successes are achieved.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<int,implicit_t<T,U>> simulate_negative_binomial(const T& k,
    const U& ρ) {
  return simulate_negative_binomial<int,T,U,int>(k, ρ);
}

/**
 * Simulate a Poisson distribution.
 *
 * @ingroup random
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param λ Rate.
 * 
 * @return Variate.
 */
template<class R, class T, class = std::enable_if_t<
    is_numeric_v<R> && is_numeric_v<T>,int>>
explicit_t<R,T> simulate_poisson(const T& λ);

/**
 * Simulate a Poisson distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * 
 * @param λ Rate.
 * 
 * @return Variate.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
explicit_t<int,T> simulate_poisson(const T& λ) {
  return simulate_poisson<int,T,int>(λ);
}

/**
 * Simulate a Student's $t$-distribution.
 *
 * @ingroup random
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param ν Degrees of freedom.
 * 
 * @return Variate.
 */
template<class R, class T, class = std::enable_if_t<
    is_floating_point_v<R> && is_numeric_v<T>,int>>
explicit_t<R,T> simulate_student_t(const T& ν);

/**
 * Simulate a Student's $t$-distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * 
 * @param ν Degrees of freedom.
 * 
 * @return Variate.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> simulate_student_t(const T& ν) {
  return simulate_student_t<value_t<default_t<T>>,T,int>(ν);
}

/**
 * Simulate a uniform distribution.
 *
 * @ingroup random
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param l Lower bound.
 * @param u Upper bound.
 * 
 * @return Variate.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,implicit_t<T,U>> simulate_uniform(const T& l, const U& u);

/**
 * Simulate a uniform distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param l Lower bound.
 * @param u Upper bound.
 * 
 * @return Variate.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
default_t<implicit_t<T,U>> simulate_uniform(const T& l, const U& u) {
  return simulate_uniform<value_t<default_t<implicit_t<T,U>>>,T,U,int>(l, u);
}

/**
 * Simulate a uniform distribution over integers.
 *
 * @ingroup random
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param l Lower bound (inclusive).
 * @param u Upper bound (inclusive).
 * 
 * @return Variate.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,implicit_t<T,U>> simulate_uniform_int(const T& l, const U& u);

/**
 * Simulate a uniform distribution over integers.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param l Lower bound (inclusive).
 * @param u Upper bound (inclusive).
 * 
 * @return Variate.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<int,implicit_t<T,U>> simulate_uniform_int(const T& l, const U& u) {
  return simulate_uniform_int<int,T,U,int>(l, u);
}

/**
 * Simulate a Weibull distribution.
 *
 * @ingroup random
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param k Shape.
 * @param λ Scale.
 * 
 * @return Variate.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,implicit_t<T,U>> simulate_weibull(const T& k, const U& λ);

/**
 * Simulate a Weibull distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param k Shape.
 * @param λ Scale.
 * 
 * @return Variate.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
default_t<implicit_t<T,U>> simulate_weibull(const T& k, const U& λ) {
  return simulate_weibull<value_t<default_t<implicit_t<T,U>>>,T,U,int>(k, λ);
}

}
