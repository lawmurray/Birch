/**
 * ile
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
 * Seed pseudorandom number generators with entropy.
 * 
 * @ingroup random
 */
void seed();

/**
 * Simulate a Bernoulli distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * 
 * @param rho Probability of success.
 * 
 * @return Variate.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
explicit_t<bool,T> simulate_bernoulli(const T& rho);

/**
 * Simulate a beta distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param alpha First shape.
 * @param beta Second shape.
 * 
 * @return Variate.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
default_t<T,U> simulate_beta(const T& alpha, const U& beta);

/**
 * Simulate a binomial distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param n Number of trials.
 * @param rho Probability of success.
 * 
 * @return Variate.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
explicit_t<int,T,U> simulate_binomial(const T& n, const U& rho);

/**
 * Simulate a $\chi^2$ distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * 
 * @param nu Degrees of freedom.
 * 
 * @return Variate.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> simulate_chi_squared(const T& nu);

/**
 * Simulate an exponential distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * 
 * @param lambda Rate.
 * 
 * @return Variate.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> simulate_exponential(const T& lambda);

/**
 * Simulate a gamma distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param k Shape.
 * @param theta Scale.
 * 
 * @return Variate.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
default_t<T,U> simulate_gamma(const T& k, const U& theta);

/**
 * Simulate a Gaussian distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param mu Mean.
 * @param sigma2 Variance.
 * 
 * @return Variate.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
default_t<T,U> simulate_gaussian(const T& mu, const U& sigma2);

/**
 * Simulate a negative binomial distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param k Number of successes.
 * @param rho Probability of success.
 * 
 * @return Number of failures before `k` number of successes are achieved.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
explicit_t<int,T,U> simulate_negative_binomial(const T& k, const U& rho);

/**
 * Simulate a Poisson distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * 
 * @param lambda Rate.
 * 
 * @return Variate.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
explicit_t<int,T> simulate_poisson(const T& lambda);

/**
 * Simulate a Student's $t$-distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * 
 * @param nu Degrees of freedom.
 * 
 * @return Variate.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> simulate_student_t(const T& nu);

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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
default_t<T,U> simulate_uniform(const T& l, const U& u);

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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
explicit_t<int,T,U> simulate_uniform_int(const T& l, const U& u);

/**
 * Simulate a Weibull distribution.
 *
 * @ingroup random
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param k Shape.
 * @param lambda Scale.
 * 
 * @return Variate.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
default_t<T,U> simulate_weibull(const T& k, const U& lambda);

/**
 * Create vector of standard Gaussian variates (mean zero, variance one).
 *
 * @ingroup random
 * 
 * @param n Number of elements.
 * 
 * @return Variates.
 */
Array<real,1> standard_gaussian(const int n);

/**
 * Create matrix of standard Gaussian variates (mean zero, variance one).
 *
 * @ingroup random
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * 
 * @return Variates.
 */
Array<real,2> standard_gaussian(const int m, const int n);

/**
 * Create matrix of standard Wishart (scale one).
 *
 * @ingroup random
 * 
 * @tparam T Scalar type.
 * 
 * @param nu Degrees of freedom.
 * @param n Number of rows and columns.
 * 
 * @return Variates.
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
Array<real,2> standard_wishart(const T& nu, const int n);

}
