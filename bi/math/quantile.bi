/**
 * Quantile of a binomial distribution.
 *
 * - p: The cumulative probability.
 * - n: Number of trials.
 * - ρ: Probability of a true result.
 *
 * Return: the quantile.
 */
function quantile_binomial(p:Real, n:Integer, ρ:Real) -> Integer {
  assert 0 <= n;
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp{{
  return boost::math::quantile(boost::math::binomial_distribution<>(n, ρ), p);
  }}
}

/**
 * Quantile of a negative binomial distribution.
 *
 * - p: The cumulative probability.
 * - k: Number of successes before the experiment is stopped.
 * - ρ: Probability of success.
 *
 * Return: the quantile.
 */
function quantile_negative_binomial(p:Real, k:Integer, ρ:Real) -> Integer {
  assert 0 < k;
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp{{
  return boost::math::quantile(boost::math::negative_binomial_distribution<>(k, ρ), p);
  }}
}

/**
 * Quantile of a Poisson distribution.
 *
 * - p: The cumulative probability.
 * - λ: Rate.
 *
 * Return: the quantile.
 */
function quantile_poisson(p:Real, λ:Real) -> Integer {
  assert 0.0 <= λ;
  cpp{{
  return boost::math::quantile(boost::math::poisson_distribution<>(λ), p);
  }}
}

/**
 * Quantile of a uniform distribution.
 *
 * - p: The cumulative probability.
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 *
 * Return: the quantile.
 */
function quantile_uniform(p:Real, l:Real, u:Real) -> Real {
  assert l <= u;
  return l + p*(u - l);
}

/**
 * Quantile of an exponential distribution.
 *
 * - p: The cumulative probability.
 * - λ: Rate.
 *
 * Return: the quantile.
 */
function quantile_exponential(p:Real, λ:Real) -> Real {
  assert 0.0 < λ;
  cpp{{
  return boost::math::quantile(boost::math::exponential_distribution<>(λ), p);
  }}
}

/**
 * Quantile of a Weibull distribution.
 *
 * - p: The cumulative probability.
 * - k: Shape.
 * - λ: Scale.
 *
 * Return: the quantile.
 */
function quantile_weibull(p:Real, k:Real, λ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < λ;
  cpp{{
  return boost::math::quantile(boost::math::weibull_distribution<>(k, λ), p);
  }}
}

/**
 * Quantile of a Gaussian distribution.
 *
 * - p: The cumulative probability.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Return: the quantile.
 */
function quantile_gaussian(p:Real, μ:Real, σ2:Real) -> Real {
  assert 0.0 < σ2;
  cpp{{
  return boost::math::quantile(boost::math::normal_distribution<>(μ, ::sqrt(σ2)), p);
  }}
}

/**
 * Quantile of a Student's $t$ distribution.
 *
 * - p: The cumulative probability.
 * - ν: Degrees of freedom.
 *
 * Return: the quantile.
 */
function quantile_student_t(p:Real, ν:Real) -> Real {
  assert 0.0 < ν;
  cpp{{
  return boost::math::quantile(boost::math::students_t_distribution<>(ν), p);
  }}
}

/**
 * Quantile of a Student's $t$ distribution with location and scale.
 *
 * - p: The cumulative probability.
 * - ν: Degrees of freedom.
 * - μ: Location.
 * - σ2: Squared scale.
 *
 * Return: the quantile.
 */
function quantile_student_t(p:Real, ν:Real, μ:Real, σ2:Real) -> Real {
  assert 0.0 < σ2;
  return quantile_student_t(p, ν)*sqrt(σ2) + μ;
}

/**
 * Quantile of a beta distribution.
 *
 * - p: The cumulative probability.
 * - α: Shape.
 * - β: Shape.
 *
 * Return: the quantile.
 */
function quantile_beta(p:Real, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;  
  cpp{{
  return boost::math::quantile(boost::math::beta_distribution<>(α, β), p);
  }}
}

/**
 * CDF of $\chi^2$ distribution.
 *
 * - p: The cumulative probability.
 * - ν: Degrees of freedom.
 *
 * Return: the quantile.
 */
function quantile_chi_squared(p:Real, ν:Real) -> Real {
  assert 0.0 < ν;
  cpp{{
  return boost::math::quantile(boost::math::chi_squared_distribution<>(ν), p);
  }}
}

/**
 * Quantile of a gamma distribution.
 *
 * - p: The cumulative probability.
 * - k: Shape.
 * - θ: Scale.
 *
 * Return: the quantile.
 */
function quantile_gamma(p:Real, k:Real, θ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < θ;
  cpp{{
  return boost::math::quantile(boost::math::gamma_distribution<>(k, θ), p);
  }}
}

/**
 * Quantile of an inverse-gamma distribution.
 *
 * - p: The cumulative probability.
 * - α: Shape.
 * - β: Scale.
 *
 * Return: the quantile.
 */
function quantile_inverse_gamma(p:Real, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;
  cpp{{
  return boost::math::quantile(boost::math::inverse_gamma_distribution<>(α, β), p);
  }}
}

/**
 * Quantile of a normal inverse-gamma distribution.
 *
 * - p: The cumulative probability.
 * - μ: Mean.
 * - a2: Variance.
 * - α: Shape of inverse-gamma on scale.
 * - β: Scale of inverse-gamma on scale.
 *
 * Return: the quantile.
 */
function quantile_normal_inverse_gamma(p:Real, μ:Real, a2:Real, α:Real,
    β:Real) -> Real {
  return quantile_student_t(p, 2.0*α, μ, a2*β/α);
}

/**
 * Quantile of a gamma-Poisson distribution.
 *
 * - p: The cumulative probability.
 * - k: Shape.
 * - θ: Scale.
 *
 * Return: the quantile.
 */
function quantile_gamma_poisson(p:Real, k:Real, θ:Real) -> Integer {
  assert 0.0 < k;
  assert 0.0 < θ;
  assert k == floor(k);
  return quantile_negative_binomial(p, Integer(k), 1.0/(θ + 1.0));
}

/**
 * Quantile of a Lomax distribution.
 *
 * - p: The cumulative probability.
 * - λ: Scale.
 * - α: Shape.
 *
 * Return: the quantile.
 */
function quantile_lomax(p:Real, λ:Real, α:Real) -> Real {
  assert 0.0 < λ;
  assert 0.0 < α;
  cpp{{
  return boost::math::quantile(boost::math::pareto_distribution<>(λ, α), p) - λ;
  }}
}

/**
 * Quantile of a Gaussian distribution with an inverse-gamma distribution over
 * the variance.
 *
 * - p: The cumulative probability.
 * - μ: Mean.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Return: the quantile.
 */
function quantile_inverse_gamma_gaussian(p:Real, μ:Real, α:Real, β:Real) -> Real {
  return quantile_student_t(p, 2.0*α, μ, β/α);
}

/**
 * Quantile of a Gaussian distribution with a normal inverse-gamma prior.
 *
 * - p: The cumulative probability.
 * - μ: Mean.
 * - a2: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Return: the quantile.
 */
function quantile_normal_inverse_gamma_gaussian(p:Real, μ:Real, a2:Real,
    α:Real, β:Real) -> Real {
  return quantile_student_t(p, 2.0*α, μ, (β/α)*(1.0 + a2));
}

/**
 * Quantile of a Gaussian distribution with a normal inverse-gamma prior with linear
 * transformation.
 *
 * - p: The cumulative probability.
 * - a: Scale.
 * - μ: Mean.
 * - c: Offset.
 * - a2: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Return: the quantile.
 */
function quantile_linear_normal_inverse_gamma_gaussian(p:Real, a:Real,
    μ:Real, c:Real, a2:Real, α:Real, β:Real) -> Real {
  return quantile_student_t(p, 2.0*α, a*μ + c, (β/α)*(1.0 + a*a*a2));
}
