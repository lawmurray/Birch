/**
 * PMF of a Bernoulli variate.
 *
 * - x: The variate.
 * - ρ: Probability of a true result.
 *
 * Return: the probability mass.
 */
function pdf_bernoulli(x:Boolean, ρ:Real) -> Real {
  assert 0.0 <= ρ && ρ <= 1.0;
  if (x) {
    return ρ;
  } else {
    return 1.0 - ρ;
  }
}

/**
 * PMF of a delta variate.
 *
 * - x: The variate.
 * - μ: Location.
 *
 * Return: the probability mass.
 */
function pdf_delta(x:Integer, μ:Integer) -> Real {
  if (x == μ) {
    return 1.0;
  } else {
    return 0.0;
  }
}

/**
 * PMF of a binomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - ρ: Probability of a true result.
 *
 * Return: the probability mass.
 */
function pdf_binomial(x:Integer, n:Integer, ρ:Real) -> Real {
  assert 0 <= n;
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp{{
  return boost::math::pdf(boost::math::binomial_distribution<>(n, ρ), x);
  }}
}

/**
 * PMF of a negative binomial variate.
 *
 * - x: The variate (number of failures).
 * - k: Number of successes before the experiment is stopped.
 * - ρ: Probability of success.
 *
 * Return: the probability mass.
 */
function pdf_negative_binomial(x:Integer, k:Integer, ρ:Real) -> Real {
  assert 0 < k;
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp{{
  return boost::math::pdf(boost::math::negative_binomial_distribution<>(k, ρ), x);
  }}
}

/**
 * PMF of a Poisson variate.
 *
 * - x: The variate.
 * - λ: Rate.
 *
 * Return: the probability mass.
 */
function pdf_poisson(x:Integer, λ:Real) -> Real {
  assert 0.0 <= λ;
  cpp{{
  return boost::math::pdf(boost::math::poisson_distribution<>(λ), x);
  }}
}

/**
 * PMF of an integer uniform variate.
 *
 * - x: The variate.
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 *
 * Return: the probability mass.
 */
function pdf_uniform_int(x:Integer, l:Integer, u:Integer) -> Real {
  if (x >= l && x <= u) {
    return 1.0/(u - l + 1);
  } else {
    return 0.0;
  }
}

/**
 * PMF of a categorical variate.
 *
 * - x: The variate.
 * - ρ: Category probabilities.
 *
 * Return: the probability mass.
 */
function pdf_categorical(x:Integer, ρ:Real[_]) -> Real {
  if (1 <= x && x <= length(ρ)) {
    assert ρ[x] >= 0.0;
    return ρ[x];
  } else {
    return 0.0;
  }
}

/**
 * PDF of a compound-gamma variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - α: Prior shape.
 * - β: Prior scale.
 *
 * Return: the probability density.
 */
function pdf_compound_gamma(x:Real, k:Real, α:Real, β:Real) -> Real {
  return exp(logpdf_compound_gamma(x, k, α, β));
}

/**
 * PMF of a multinomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - ρ: Category probabilities.
 *
 * Return: the probability mass.
 */
function pdf_multinomial(x:Integer[_], n:Integer, ρ:Real[_]) -> Real {
  return exp(logpdf_multinomial(x, n, ρ));
}

/**
 * PDF of a Dirichlet variate.
 *
 * - x: The variate.
 * - α: Concentrations.
 *
 * Return: the probability density.
 */
function pdf_dirichlet(x:Real[_], α:Real[_]) -> Real {
  return exp(logpdf_dirichlet(x, α));
}

/**
 * PDF of a uniform variate.
 *
 * - x: The variate.
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 *
 * Return: the probability density.
 */
function pdf_uniform(x:Real, l:Real, u:Real) -> Real {
  assert l <= u;

  if (x >= l && x <= u) {
    return 1.0/(u - l);
  } else {
    return 0.0;
  }
}

/**
 * PDF of an exponential variate.
 *
 * - x: The variate.
 * - λ: Rate.
 *
 * Return: the probability density.
 */
function pdf_exponential(x:Real, λ:Real) -> Real {
  assert 0.0 < λ;
  cpp{{
  return boost::math::pdf(boost::math::exponential_distribution<>(λ), x);
  }}
}

/**
 * PDF of a Weibull variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - λ: Scale.
 *
 * Return: the probability density.
 */
function pdf_weibull(x:Real, k:Real, λ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < λ;
  cpp{{
  return boost::math::pdf(boost::math::weibull_distribution<>(k, λ), x);
  }}
}

/**
 * PDF of a Gaussian variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Return: the probability density.
 */
function pdf_gaussian(x:Real, μ:Real, σ2:Real) -> Real {
  assert 0.0 < σ2;
  cpp{{
  return boost::math::pdf(boost::math::normal_distribution<>(μ, ::sqrt(σ2)), x);
  }}
}

/**
 * PDF of a Student's $t$ variate.
 *
 * - x: The variate.
 * - ν: Degrees of freedom.
 *
 * Return: the probability density.
 */
function pdf_student_t(x:Real, ν:Real) -> Real {
  assert 0.0 < ν;
  cpp{{
  return boost::math::pdf(boost::math::students_t_distribution<>(ν), x);
  }}
}

/**
 * PDF of a Student's $t$ variate with location and scale.
 *
 * - x: The variate.
 * - ν: Degrees of freedom.
 * - μ: Location.
 * - σ2: Squared scale.
 *
 * Return: the probability density.
 */
function pdf_student_t(x:Real, ν:Real, μ:Real, σ2:Real) -> Real {
  assert 0.0 < ν;
  assert 0.0 < σ2;
  σ:Real <- sqrt(σ2);
  return pdf_student_t((x - μ)/σ, ν)/σ;
}

/**
 * PDF of a beta variate.
 *
 * - x: The variate.
 * - α: Shape.
 * - β: Shape.
 *
 * Return: the probability density.
 */
function pdf_beta(x:Real, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;
  cpp{{
  return boost::math::pdf(boost::math::beta_distribution<>(α, β), x);
  }}
}

/**
 * PDF of $\chi^2$ variate.
 *
 * - x: The variate.
 * - ν: Degrees of freedom.
 *
 * Return: the probability density.
 */
function pdf_chi_squared(x:Real, ν:Real) -> Real {
  assert 0.0 < ν;
  cpp{{
  return boost::math::pdf(boost::math::chi_squared_distribution<>(ν), x);
  }}
}

/**
 * PDF of a gamma variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - θ: Scale.
 *
 * Return: the probability density.
 */
function pdf_gamma(x:Real, k:Real, θ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < θ;
  cpp{{
  return boost::math::pdf(boost::math::gamma_distribution<>(k, θ), x);
  }}
}

/**
 * PDF of a Wishart variate.
 *
 * - X: The variate.
 * - Ψ: Scale.
 * - ν: Degrees of freedom.
 *
 * Returns: the probability density.
 */
function pdf_wishart(X:Real[_,_], Ψ:Real[_,_], ν:Real) -> Real {
  return exp(logpdf_wishart(X, Ψ, ν));
}

/**
 * PDF of an inverse-gamma variate.
 *
 * - x: The variate.
 * - α: Shape.
 * - β: Scale.
 *
 * Return: the probability density.
 */
function pdf_inverse_gamma(x:Real, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;
  cpp{{
  return boost::math::pdf(boost::math::inverse_gamma_distribution<>(α, β), x);
  }}
}

/**
 * PDF of an inverse Wishart variate.
 *
 * - X: The variate.
 * - Ψ: Scale.
 * - ν: Degrees of freedom.
 *
 * Returns: the probability density.
 */
function pdf_inverse_wishart(X:Real[_,_], Ψ:Real[_,_], ν:Real) -> Real {
  return exp(logpdf_inverse_wishart(X, Ψ, ν));
}

/**
 * PDF of a normal inverse-gamma variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - a2: Variance.
 * - α: Shape of inverse-gamma on scale.
 * - β: Scale of inverse-gamma on scale.
 *
 * Return: the probability density.
 */
function pdf_normal_inverse_gamma(x:Real, μ:Real, a2:Real, α:Real,
    β:Real) -> Real {
  return pdf_student_t(x, 2.0*α, μ, a2*β/α);
}

/**
 * PMF of a beta-bernoulli variate.
 *
 * - x: The variate.
 * - α: Shape.
 * - β: Shape.
 *
 * Return: the probability mass.
 */
function pdf_beta_bernoulli(x:Boolean, α:Real, β:Real) -> Real {
  return exp(logpdf_beta_bernoulli(x, α, β));
}

/**
 * PMF of a beta-binomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - α: Shape.
 * - β: Shape.
 *
 * Return: the probability mass.
 */
function pdf_beta_binomial(x:Integer, n:Integer, α:Real, β:Real) -> Real {
  return exp(logpdf_beta_binomial(x, n, α, β));
}

/**
 * PMF of a beta-negative-binomial variate.
 *
 * - x: The variate.
 * - n: Number of successes.
 * - α: Shape.
 * - β: Shape.
 *
 * Return: the probability mass.
 */
function pdf_beta_negative_binomial(x:Integer, k:Integer, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;
  assert 0 < x;

  return exp(logpdf_beta_negative_binomial(x, k, α, β));
}

/**
 * PMF of a gamma-Poisson variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - θ: Scale.
 *
 * Return: the probability mass.
 */
function pdf_gamma_poisson(x:Integer, k:Real, θ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < θ;
  assert k == floor(k);

  return pdf_negative_binomial(x, Integer(k), 1.0/(θ + 1.0));
}

/**
 * PDF of a Lomax variate.
 *
 * - x: The variate.
 * - λ: Scale.
 * - α: Shape.
 *
 * Return: the probability density.
 */
function pdf_lomax(x:Real, λ:Real, α:Real) -> Real {
  assert 0.0 < λ;
  assert 0.0 < α;

  cpp{{
  return boost::math::pdf(boost::math::pareto_distribution<>(λ, α), x + λ);
  }}
}

/**
 * PMF of a Dirichlet-categorical variate.
 *
 * - x: The variate.
 * - α: Concentrations.
 *
 * Return: the probability mass.
 */
function pdf_dirichlet_categorical(x:Integer, α:Real[_]) -> Real {
  return exp(logpdf_dirichlet_categorical(x, α));
}

/**
 * PMF of a Dirichlet-multinomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - α: Concentrations.
 *
 * Return: the probability mass.
 */
function pdf_dirichlet_multinomial(x:Integer[_], n:Integer, α:Real[_]) -> Real {
  return exp(logpdf_dirichlet_multinomial(x, n, α));
}

/**
 * PMF of a categorical variate with Chinese restaurant process prior.
 *
 * - x: The variate.
 * - α: Concentration.
 * - θ: Discount.
 * - n: Enumerated items.
 * - N: Total number of items.
 *
 * Return: the probability mass.
 */
function pdf_restaurant_categorical(k:Integer, α:Real, θ:Real,
    n:Integer[_], N:Integer) -> Real {
  K:Integer <- length(n);
  if (k > K + 1) {
    return 0.0;
  } else if (k == K + 1) {
    return (K*α + θ)/(N + θ);
  } else {
    return (n[k] - α)/(N + θ);
  }
}

/**
 * PDF of a Gaussian variate with an inverse-gamma distribution over
 * the variance.
 *
 * - x: The variate.
 * - μ: Mean.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Return: the probability density.
 */
function pdf_inverse_gamma_gaussian(x:Real, μ:Real, α:Real,
    β:Real) -> Real {
  return pdf_student_t(x, 2.0*α, μ, β/α);
}

/**
 * PDF of a Gaussian variate with a normal inverse-gamma prior.
 *
 * - x: The variate.
 * - μ: Mean.
 * - a2: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Return: the probability density.
 */
function pdf_normal_inverse_gamma_gaussian(x:Real, μ:Real, a2:Real,
    α:Real, β:Real) -> Real {
  return pdf_student_t(x, 2.0*α, μ, (β/α)*(1.0 + a2));
}

/**
 * PDF of a Gaussian variate with a normal inverse-gamma prior with linear
 * transformation.
 *
 * - x: The variate.
 * - a: Scale.
 * - μ: Mean.
 * - c: Offset.
 * - a2: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Return: the probability density.
 */
function pdf_linear_normal_inverse_gamma_gaussian(x:Real, a:Real,
    μ:Real, c:Real, a2:Real, α:Real, β:Real) -> Real {
  return pdf_student_t(x, 2.0*α, a*μ + c, (β/α)*(1.0 + a*a*a2));
}

/**
 * PDF of a multivariate Gaussian variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - Σ: Covariance.
 *
 * Return: the probability density.
 */
function pdf_multivariate_gaussian(x:Real[_], μ:Real[_], Σ:Real[_,_]) ->
    Real {
  return exp(logpdf_multivariate_gaussian(x, μ, Σ));
}

/**
 * PDF of a multivariate Gaussian distribution with independent elements.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Return: the probability density.
 */
function pdf_multivariate_gaussian(x:Real[_], μ:Real[_], σ2:Real[_]) -> Real {
  return exp(logpdf_multivariate_gaussian(x, μ, σ2));
}

/**
 * PDF of a multivariate Gaussian distribution with independent elements of
 * identical variance.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Return: the probability density.
 */
function pdf_multivariate_gaussian(x:Real[_], μ:Real[_], σ2:Real) -> Real {
  return exp(logpdf_multivariate_gaussian(x, μ, σ2));
}

/**
 * PDF of a multivariate Gaussian variate with an inverse-gamma distribution
 * over a diagonal covariance.
 *
 * - x: The variate.
 * - μ: Mean.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Return: the probability density.
 */
function pdf_inverse_gamma_multivariate_gaussian(x:Real[_], μ:Real[_],
    α:Real, β:Real) -> Real {
  return pdf_multivariate_student_t(x, 2.0*α, μ, β/α);
}

/**
 * PDF of a multivariate normal inverse-gamma variate.
 *
 * - x: The variate.
 * - ν: Precision times mean.
 * - Λ: Precision.
 * - α: Shape of inverse-gamma on scale.
 * - β: Scale of inverse-gamma on scale.
 *
 * Return: the probability density.
 */
function pdf_multivariate_normal_inverse_gamma(x:Real[_], ν:Real[_],
    Λ:LLT, α:Real, β:Real) -> Real {
  return pdf_multivariate_student_t(x, 2.0*α, solve(Λ, ν), (β/α)*inv(Λ));
}

/**
 * PDF of a multivariate Gaussian variate with a multivariate normal
 * inverse-gamma prior.
 *
 * - x: The variate.
 * - ν: Precision times mean.
 * - Λ: Precision.
 * - α: Shape of the inverse-gamma.
 * - γ: Scale accumulator of the inverse-gamma.
 *
 * Return: the probability density.
 */
function pdf_multivariate_normal_inverse_gamma_multivariate_gaussian(x:Real[_], ν:Real[_],
    Λ:LLT, α:Real, γ:Real) -> Real {
  return exp(logpdf_multivariate_normal_inverse_gamma_multivariate_gaussian(
      x, ν, Λ, α, γ));
}

/**
 * PDF of a multivariate Gaussian variate with a multivariate normal
 * inverse-gamma prior with linear transformation.
 *
 * - x: The variate.
 * - A: Scale.
 * - ν: Precision times mean.
 * - c: Offset.
 * - Λ: Precision.
 * - α: Shape of the inverse-gamma.
 * - γ: Scale accumulator of the inverse-gamma.
 *
 * Return: the probability density.
 */
function pdf_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(x:Real[_],
    A:Real[_,_], ν:Real[_], c:Real[_], Λ:LLT, α:Real, γ:Real) -> Real {
  return exp(logpdf_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
      x, A, ν, c, Λ, α, γ));
}

/**
 * PDF of a matrix Gaussian distribution.
 *
 * - X: The variate.
 * - M: Mean.
 * - U: Among-row covariance.
 * - V: Among-column covariance.
 *
 * Returns: the probability density.
 */
function pdf_matrix_gaussian(X:Real[_,_], M:Real[_,_], U:Real[_,_],
    V:Real[_,_]) -> Real {
  return exp(logpdf_matrix_gaussian(X, M, U, V));
}

/**
 * PDF of a matrix normal-inverse-gamma variate.
 *
 * - X: The variate.
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - α: Variance shape.
 * - γ: Variance scale accumulators.
 *
 * Returns: the probability density.
 */
function pdf_matrix_normal_inverse_gamma(X:Real[_,_], N:Real[_,_], Λ:LLT,
    α:Real, γ:Real[_]) -> Real {
  return exp(logpdf_matrix_normal_inverse_gamma(X, N, Λ, α, γ));
}

/**
 * PDF of a Gaussian variate with matrix normal inverse-gamma prior.
 *
 * - X: The variate.
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - α: Variance shape.
 * - γ: Variance scale accumulators.
 *
 * Returns: the probability density.
 */
function pdf_matrix_normal_inverse_gamma_matrix_gaussian(
    X:Real[_,_], N:Real[_,_], Λ:LLT, α:Real, γ:Real[_]) -> Real {
  return exp(logpdf_matrix_normal_inverse_gamma_matrix_gaussian(X,
      N, Λ, α, γ));
}

/**
 * PDF of a Gaussian variate with matrix normal inverse-gamma prior.
 *
 * - X: The variate.
 * - A: Scale.
 * - N: Precision times mean matrix.
 * - C: Offset.
 * - Λ: Precision.
 * - α: Variance shape.
 * - γ: Variance scale accumulators.
 *
 * Returns: the probability density.
 */
function pdf_linear_matrix_normal_inverse_gamma_matrix_gaussian(
    X:Real[_,_], A:Real[_,_], N:Real[_,_], C:Real[_,_], Λ:LLT, α:Real,
    γ:Real[_]) -> Real {
  return exp(logpdf_linear_matrix_normal_inverse_gamma_matrix_gaussian(X,
      A, N, C, Λ, α, γ));
}

/**
 * PDF of a matrix normal-inverse-Wishart variate.
 *
 * - X: The variate.
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - V: Prior variance shape.
 * - k: Prior degrees of freedom.
 *
 * Returns: the probability density.
 */
function pdf_matrix_normal_inverse_wishart(X:Real[_,_], N:Real[_,_],
    Λ:LLT, V:Real[_,_], k:Real) -> Real {
  return exp(logpdf_matrix_normal_inverse_wishart(X, N, Λ, V, k));
}

/**
 * PDF of a Gaussian variate with matrix-normal-inverse-Wishart prior.
 *
 * - X: The variate.
 * - N: Prior precision times mean matrix.
 * - Λ: Prior precision.
 * - V: Prior variance shape.
 * - k: Prior degrees of freedom.
 *
 * Returns: the probability density.
 */
function pdf_matrix_normal_inverse_wishart_matrix_gaussian(X:Real[_,_],
    N:Real[_,_], Λ:LLT, V:Real[_,_], k:Real) -> Real {
  return exp(logpdf_matrix_normal_inverse_wishart_matrix_gaussian(X, N, Λ, V,
      k));
}

/**
 * PDF of a Gaussian variate with linear transformation of a
 * matrix-normal-inverse-Wishart prior.
 *
 * - X: The variate.
 * - A: Scale.
 * - N: Prior precision times mean matrix.
 * - C: Offset.
 * - Λ: Prior precision.
 * - V: Prior variance shape.
 * - k: Prior degrees of freedom.
 *
 * Returns: the probability density.
 */
function pdf_linear_matrix_normal_inverse_wishart_matrix_gaussian(
    X:Real[_,_], A:Real[_,_], N:Real[_,_], C:Real[_,_], Λ:LLT, V:Real[_,_],
    k:Real) -> Real {
  return exp(logpdf_linear_matrix_normal_inverse_wishart_matrix_gaussian(X,
      A, N, C, Λ, V, k));
}

/**
 * PDF of a multivariate Student's $t$-distribution variate with location
 * and scale.
 *
 * - x: The variate.
 * - k: Degrees of freedom.
 * - μ: Mean.
 * - Σ: Covariance.
 *
 * Return: the probability density.
 */
function pdf_multivariate_student_t(x:Real[_], k:Real, μ:Real[_], Σ:Real[_,_]) ->
    Real {
  return exp(logpdf_multivariate_student_t(x, k, μ, Σ));
}

/**
 * PDF of a multivariate Student's $t$-distribution variate with location
 * and diagonal scale.
 *
 * - x: The variate.
 * - k: Degrees of freedom.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Return: the probability density.
 */
function pdf_multivariate_student_t(x:Real[_], k:Real, μ:Real[_],
    σ2:Real) -> Real {
  return exp(logpdf_multivariate_student_t(x, k, μ, σ2));
}

/**
 * PDF of a matrix Student's $t$-distribution variate with location
 * and scale.
 *
 * - X: The variate.
 * - k: Degrees of freedom.
 * - M: Mean.
 * - U: Among-row covariance.
 * - V: Among-column covariance.
 *
 * Returns: the probability density.
 */
function pdf_matrix_student_t(X:Real[_,_], k:Real, M:Real[_,_], U:Real[_,_],
    V:Real[_,_]) -> Real {
  return exp(logpdf_matrix_student_t(X, k, M, U, V));
}

/**
 * PDF of a matrix Student's $t$-distribution variate with location
 * and diagonal scale.
 *
 * - X: The variate.
 * - k: Degrees of freedom.
 * - M: Mean.
 * - U: Among-row covariance.
 * - v: Independent within-column covariance.
 *
 * Returns: the probability density.
 */
function pdf_matrix_student_t(X:Real[_,_], k:Real, M:Real[_,_], U:Real[_,_],
    v:Real[_]) -> Real {
  return exp(logpdf_matrix_student_t(X, k, M, U, v));
}

/**
 * PDF of a multivariate uniform variate.
 *
 * - x: The variate.
 * - l: Lower bound of hyperrectangle.
 * - u: Upper bound of hyperrectangle.
 *
 * Return: the probability density.
 */
function pdf_independent_uniform(x:Real[_], l:Real[_], u:Real[_]) -> Real {
  assert length(x) > 0;
  assert length(l) == length(x);
  assert length(u) == length(x);
  
  D:Integer <- length(x);
  w:Real <- 1.0;
  for (d:Integer in 1..D) {
    w <- w*pdf_uniform(x[d], l[d], u[d]);
  }
  return w;
}

/**
 * PMF of a multivariate integer uniform variate.
 *
 * - x: The variate.
 * - l: Lower bound of hyperrectangle.
 * - u: Upper bound of hyperrectangle.
 *
 * Return: the probability density.
 */
function pdf_independent_uniform_int(x:Integer[_], l:Integer[_],
    u:Integer[_]) -> Real {
  assert length(x) > 0;
  assert length(l) == length(x);
  assert length(u) == length(x);
  
  D:Integer <- length(x);
  w:Real <- 1.0;
  for (d:Integer in 1..D) {
    w <- w*pdf_uniform_int(x[d], l[d], u[d]);
  }
  return w;
}
