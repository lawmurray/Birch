cpp{{
#include "boost/math/distributions.hpp"
}}

/**
 * Log partition function of a Bernoulli variate.
 *
 * - ρ: Probability of a true result.
 *
 * Return: the log of the normalization constant.
 */
function lpartition_bernoulli(ρ:Real) -> Real {
  assert 0.0 <= ρ && ρ <= 1.0;
  return -log(1- ρ);
}

/**
 * Partition function of a Bernoulli variate.
 *
 * - ρ: Probability of a true result.
 *
 * Return: the normalization constant.
 */
function partition_bernoulli(ρ:Real) -> Real {
  assert 0.0 <= ρ && ρ <= 1.0;
  return 1/(1-ρ);
}


/**
 * Log partition function of a Binomial variate.
 *
 * - n: Number of trials.
 * - ρ: Probability of a true result.
 *
 * Return: the log of the normalization constant.
 */
function lpartition_binomial(n:Integer, ρ:Real) -> Real {
  assert 0 <= n;
  assert 0.0 <= ρ && ρ <= 1.0;
  return -n*log(1- ρ);
}

/**
 * Partition function of a Binomial variate.
 *
 * - n: Number of trials.
 * - ρ: Probability of a true result.
 *
 * Return: the normalization constant.
 */
function partition_binomial(n:Integer, ρ:Real) -> Real {
  assert 0 <= n;
  assert 0.0 <= ρ && ρ <= 1.0;
  return exp(lpartition_binomial(n, ρ));
}

/**
 * Log partition function of a negative binomial variate.
 *
 * - k: Number of successes before the experiment is stopped.
 * - ρ: Probability of success.
 *
 * Return: the log of the normalization constant.
 */
function lpartition_negative_binomial(k:Integer, ρ:Real) -> Real {
  assert 0 < k;
  assert 0.0 <= ρ && ρ <= 1.0;
  return -k*log(1 - ρ);
}

/**
 * Partition function of a negative binomial variate.
 *
 * - k: Number of successes before the experiment is stopped.
 * - ρ: Probability of success.
 *
 * Return: the normalization constant.
 */
function partition_negative_binomial(k:Integer, ρ:Real) -> Real {
  assert 0 < k;
  assert 0.0 <= ρ && ρ <= 1.0;
  return exp(lpartition_negative_binomial(k, ρ));
}

/**
 * Log partition function of a Poisson variate.
 *
 * - λ: Rate.
 *
 * Return: the log of the normalization constant.
 */
function lpartition_poisson(λ:Real) -> Real {
  assert 0.0 <= λ;
  return λ;
}

/**
 * Partition function of a Poisson variate.
 *
 * - λ: Rate.
 *
 * Return: the normalization constant.
 */
function partition_poisson(λ:Real) -> Real {
  assert 0.0 <= λ;
  return exp(λ);
}

/**
 * Log partition function of an exponential variate.
 *
 * - λ: Rate.
 *
 * Return: the log of the normalization constant.
 */
function lpartition_exponential(λ:Real) -> Real {
  assert 0.0 < λ;
  return -log(λ);
}

/**
 * Partition function of an exponential variate.
 *
 * - λ: Rate.
 *
 * Return: the normalization constant.
 */
function partition_exponential(λ:Real) -> Real {
  assert 0.0 < λ;
  return 1/λ;
}

/**
 * Log partition function of a Weibull variate.
 *
 * - k: Shape.
 * - λ: Scale.
 *
 * Return: the log of the normalization constant.
 */
function lpartition_weibull(k:Real, λ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < λ;
  return k*log(λ) - log(k);
}

/**
 * Partition function of a Weibull variate.
 *
 * - k: Shape.
 * - λ: Scale.
 *
 * Return: the normalization constant.
 */
function partition_weibull(k:Real, λ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < λ;
  return exp(lpartition_weibull(k, λ));
}


/**
 * Log partition function of a Laplace variate.
 *
 * - b: Shape.
 *
 * Return: the log of the normalization constant.
 */
function lpartition_laplace(b:Real) -> Real {
  assert 0.0 < b;
  return log(2*b);
}

/**
 * Partition function of a Laplace variate.
 *
 * - b: Shape.
 *
 * Return: the normalization constant.
 */
function partition_laplace(b:Real) -> Real {
  assert 0.0 < b;
  return 2*b;
}

/**
 * Log partition function of a chi-squared variate.
 *
 * - ν: Degrees of freedom.
 *
 * Return: the log of the normalization constant.
 */
function lpartition_chi_squared(ν:Real) -> Real {
  return lgamma(ν/2) + log(2)*ν/2;
} 

/**
 * Partition function of a chi-squared variate.
 *
 * - ν: Degrees of freedom.
 *
 * Return: the normalization constant.
 */
function partition_chi_squared(ν:Real) -> Real {
  return exp(lpartition_chi_squared(ν));
} 

/**
 * Log partition function of a beta variate.
 *
 * α: Shape.
 * β: Shape.
 *
 * Return: the log of the normalization constant.
 */
function lpartition_beta(α:Real, β:Real) -> Real {
  return lgamma(α) + lgamma(β) - lgamma(α + β);
}

/**
 * Partition function of a beta variate.
 *
 * α: Shape.
 * β: Shape.
 *
 * Return: the normalization constant.
 */
function partition_beta(α:Real, β:Real) -> Real {
  return exp(lpartition_beta(α, β));
}
