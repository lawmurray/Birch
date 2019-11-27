/**
 * CDF of a binomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - ρ: Probability of a true result.
 *
 * Return: the cumulative probability.
 */
function cdf_binomial(x:Integer, n:Integer, ρ:Real) -> Real {
  assert 0 <= n;
  assert 0.0 <= ρ && ρ <= 1.0;
  
  if x < 0 {
    return 0.0;
  } else if x > n {
    return 1.0;
  } else {
    cpp{{
    return boost::math::cdf(boost::math::binomial_distribution<>(n, ρ), x);
    }}
  } 
}

/**
 * CDF of a negative binomial variate.
 *
 * - x: The variate (number of failures).
 * - k: Number of successes before the experiment is stopped.
 * - ρ: Probability of success.
 *
 * Return: the cumulative probability.
 */
function cdf_negative_binomial(x:Integer, k:Integer, ρ:Real) -> Real {
  assert 0 < k;
  assert 0.0 <= ρ && ρ <= 1.0;
  
  if x < 0 {
    return 0.0;
  } else {
    cpp{{
    return boost::math::cdf(boost::math::negative_binomial_distribution<>(k, ρ), x);
    }}
  }
}

/**
 * CDF of a Poisson variate.
 *
 * - x: The variate.
 * - λ: Rate.
 *
 * Return: the cumulative probability.
 */
function cdf_poisson(x:Integer, λ:Real) -> Real {
  assert 0.0 <= λ;
  
  if x < 0 {
    return 0.0;
  } else {
    cpp{{
    return boost::math::cdf(boost::math::poisson_distribution<>(λ), x);
    }}
  }
}

/**
 * CDF of a uniform integer variate.
 *
 * - x: The variate.
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 *
 * Return: the cumulative probability.
 */
function cdf_uniform_int(x:Integer, l:Integer, u:Integer) -> Real {
  if x < l {
    return 0.0;
  } else if (x > u) {
    return 1.0;
  } else {
    return (x - l + 1)/Real(u - l + 1);
  }
}

/**
 * CDF of a categorical variate.
 *
 * - x: The variate.
 * - ρ: Category probabilities.
 *
 * Return: the cumulative probability.
 */
function cdf_categorical(x:Integer, ρ:Real[_]) -> Real {
  if 1 <= x && x <= length(ρ) {
    return sum(ρ[1..x]);
  } else {
    return -inf;
  }
}

/**
 * CDF of a uniform variate.
 *
 * - x: The variate.
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 *
 * Return: the cumulative probability.
 */
function cdf_uniform(x:Real, l:Real, u:Real) -> Real {
  assert l <= u;
  
  if x <= l {
    return 0.0;
  } else if x > u {
    return 1.0;
  } else {
    return (x - l)/(u - l);
  }
}

/**
 * CDF of a compound-gamma variate.
 *
 * - x: The variate.
 * - k: The shape.
 * - α: The prior shape.
 * - β: The prior scale.
 *
 * Return: the cumulative probability.
 */
function cdf_inverse_gamma_gamma(x:Real, k:Real, α:Real, β:Real) -> Real {
  if x <= 0.0 {
    return 0.0;
  } else {
    return ibeta(k, α, x/(β + x));
  }
}

/**
 * CDF of an exponential variate.
 *
 * - x: The variate.
 * - λ: Rate.
 *
 * Return: the cumulative probability.
 */
function cdf_exponential(x:Real, λ:Real) -> Real {
  assert 0.0 < λ;
  
  if x <= 0.0 {
    return 0.0;
  } else {
    cpp{{
    return boost::math::cdf(boost::math::exponential_distribution<>(λ), x);
    }}
  }
}

/**
 * CDF of a Weibull variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - λ: Scale.
 *
 * Return: the cumulative probability.
 */
function cdf_weibull(x:Real, k:Real, λ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < λ;
  cpp{{
  return boost::math::cdf(boost::math::weibull_distribution<>(k, λ), x);
  }}
}

/**
 * CDF of a Gaussian variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Return: the cumulative probability.
 */
function cdf_gaussian(x:Real, μ:Real, σ2:Real) -> Real {
  assert 0.0 < σ2;
  cpp{{
  return boost::math::cdf(boost::math::normal_distribution<>(μ, ::sqrt(σ2)), x);
  }}
}

/**
 * CDF of a Student's $t$ variate.
 *
 * - x: The variate.
 * - ν: Degrees of freedom.
 *
 * Return: the cumulative probability.
 */
function cdf_student_t(x:Real, ν:Real) -> Real {
  assert 0.0 < ν;
  cpp{{
  return boost::math::cdf(boost::math::students_t_distribution<>(ν), x);
  }}
}

/**
 * CDF of a Student's $t$ variate with location and scale.
 *
 * - x: The variate.
 * - ν: Degrees of freedom.
 * - μ: Location.
 * - σ2: Squared scale.
 *
 * Return: the cumulative probability.
 */
function cdf_student_t(x:Real, ν:Real, μ:Real, σ2:Real) -> Real {
  assert 0.0 < σ2;
  return cdf_student_t((x - μ)/sqrt(σ2), ν);
}

/**
 * CDF of a beta variate.
 *
 * - x: The variate.
 * - α: Shape.
 * - β: Shape.
 *
 * Return: the cumulative probability.
 */
function cdf_beta(x:Real, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;
  
  if x < 0.0 {
    return 0.0;
  } else if x > 1.0 {
    return 1.0;
  } else {
    cpp{{
    return boost::math::cdf(boost::math::beta_distribution<>(α, β), x);
    }}
  }
}

/**
 * CDF of $\chi^2$ variate.
 *
 * - x: The variate.
 * - ν: Degrees of freedom.
 *
 * Return: the cumulative probability.
 */
function cdf_chi_squared(x:Real, ν:Real) -> Real {
  assert 0.0 < ν;
  cpp{{
  return boost::math::cdf(boost::math::chi_squared_distribution<>(ν), x);
  }}
}

/**
 * CDF of a gamma variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - θ: Scale.
 *
 * Return: the cumulative probability.
 */
function cdf_gamma(x:Real, k:Real, θ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < θ;
  
  if x <= 0.0 {
    return 0.0;
  } else {
    cpp{{
    return boost::math::cdf(boost::math::gamma_distribution<>(k, θ), x);
    }}
  }
}

/**
 * CDF of an inverse-gamma variate.
 *
 * - x: The variate.
 * - α: Shape.
 * - β: Scale.
 *
 * Return: the cumulative probability.
 */
function cdf_inverse_gamma(x:Real, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;
  
  if x <= 0.0 {
    return 0.0;
  } else {
    cpp{{
    return boost::math::cdf(boost::math::inverse_gamma_distribution<>(α, β), x);
    }}
  }
}

/**
 * CDF of a normal inverse-gamma variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 * - α: Shape of inverse-gamma on scale.
 * - β: Scale of inverse-gamma on scale.
 *
 * Return: the cumulative probability.
 */
function cdf_normal_inverse_gamma(x:Real, μ:Real, σ2:Real, α:Real,
    β:Real) -> Real {
  return cdf_student_t(x, 2.0*α, μ, σ2*β/α);
}

/**
 * CDF of a beta-binomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - α: Shape.
 * - β: Shape.
 *
 * Return: the cumulative probability.
 */
function cdf_beta_binomial(x:Integer, n:Integer, α:Real, β:Real) -> Real {
  P:Real <- 0.0;
  for i in 0..min(n, x) {
    P <- P + exp(logpdf_beta_binomial(i, n, α, β));
  }
  return P;
}

/**
 * CDF of a gamma-Poisson variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - θ: Scale.
 *
 * Return: the cumulative probability.
 */
function cdf_gamma_poisson(x:Integer, k:Real, θ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < θ;
  assert k == floor(k);

  return cdf_negative_binomial(x, Integer(k), 1.0/(θ + 1.0));
}

/**
 * CDF of a Lomax variate.
 *
 * - x: The variate.
 * - λ: Scale.
 * - α: Shape.
 *
 * Return: the cumulative probability.
 */
function cdf_lomax(x:Real, λ:Real, α:Real) -> Real {
  assert 0.0 < λ;
  assert 0.0 < α;

  if x <= 0.0 {
    return 0.0;
  } else {
    cpp{{
    return boost::math::cdf(boost::math::pareto_distribution<>(λ, α), x + λ);
    }}
  }
}

/**
 * CDF of a Gaussian variate with a normal inverse-gamma prior.
 *
 * - x: The variate.
 * - μ: Mean.
 * - a2: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Return: the cumulative probability.
 */
function cdf_normal_inverse_gamma_gaussian(x:Real, μ:Real, a2:Real,
    α:Real, β:Real) -> Real {
  return cdf_student_t(x, 2.0*α, μ, (β/α)*(1.0 + a2));
}

/**
 * CDF of a Gaussian variate with a normal inverse-gamma prior with linear
 * transformation.
 *
 * - x: The variate.
 * - a: Scale.
 * - μ: Mean.
 * - a2: Variance.
 * - c: Offset.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Return: the cumulative probability.
 */
function cdf_linear_normal_inverse_gamma_gaussian(x:Real, a:Real,
    μ:Real, a2:Real, c:Real, α:Real, β:Real) -> Real {
  return cdf_student_t(x, 2.0*α, a*μ + c, (β/α)*(1.0 + a*a*a2));
}
