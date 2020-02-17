/**
 * Observe a Bernoulli variate.
 *
 * - x: The variate.
 * - ρ: Probability of a true result.
 *
 * Returns: the log probability mass.
 */
function logpdf_bernoulli(x:Boolean, ρ:Real) -> Real {
  assert 0.0 <= ρ && ρ <= 1.0;
  if (x) {
    return log(ρ);
  } else {
    return log1p(-ρ);
  }
}

/**
 * Observe a delta variate.
 *
 * - x: The variate.
 * - μ: Location.
 *
 * Returns: the log probability mass.
 */
function logpdf_delta(x:Integer, μ:Integer) -> Real {
  if x == μ {
    return 0.0;
  } else {
    return -inf;
  }
}

/**
 * Observe a binomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - ρ: Probability of a true result.
 *
 * Returns: the log probability mass.
 */
function logpdf_binomial(x:Integer, n:Integer, ρ:Real) -> Real {
  assert 0 <= n;
  assert 0.0 <= ρ && ρ <= 1.0;

  if ρ == 0.0 || ρ == 1.0 {
    if x == n*ρ {
      return 0.0;
    } else {
      return -inf;
    }
  } else if 0 <= x && x <= n {
    return x*log(ρ) + (n - x)*log1p(-ρ) + lchoose(n, x);
  } else {
    return -inf;
  }
}

/**
 * Observe a negative binomial variate.
 *
 * - x: The variate (number of failures).
 * - k: Number of successes before the experiment is stopped.
 * - ρ: Probability of success.
 *
 * Returns: the log probability mass.
 */
function logpdf_negative_binomial(x:Integer, k:Integer, ρ:Real) -> Real {
  assert 0 < k;
  assert 0.0 <= ρ && ρ <= 1.0;

  if (x >= 0) {
    return k*log(ρ) + x*log1p(-ρ) + lchoose(x + k - 1, x);
  } else {
    return -inf;
  }
}

/**
 * Observe a Poisson variate.
 *
 * - x: The variate.
 * - λ: Rate.
 *
 * Returns: the log probability mass.
 */
function logpdf_poisson(x:Integer, λ:Real) -> Real {
  assert 0.0 <= λ;

  if (λ > 0.0) {
    if (x >= 0) {
      return x*log(λ) - λ - lgamma(x + 1.0);
    } else {
      return -inf;
    }
  } else {
    if (x == 0) {
      return inf;
    } else {
      return -inf;
    }
  }
}

/**
 * Observe an integer uniform variate.
 *
 * - x: The variate.
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 *
 * Returns: the log probability mass.
 */
function logpdf_uniform_int(x:Integer, l:Integer, u:Integer) -> Real {
  if (x >= l && x <= u) {
    return -log1p(Real(u - l));
  } else {
    return -inf;
  }
}

/**
 * Observe a categorical variate.
 *
 * - x: The variate.
 * - ρ: Category probabilities.
 *
 * Returns: the log probability mass.
 */
function logpdf_categorical(x:Integer, ρ:Real[_]) -> Real {
  if (1 <= x && x <= length(ρ)) {
    assert ρ[x] >= 0.0;
    return log(ρ[x]);
  } else {
    return -inf;
  }
}

/**
 * Observe a multinomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - ρ: Category probabilities.
 *
 * Returns: the log probability mass.
 */
function logpdf_multinomial(x:Integer[_], n:Integer, ρ:Real[_]) -> Real {
  assert length(x) == length(ρ);

  m:Integer <- 0;
  w:Real <- lgamma(n + 1.0);
  for i in 1..length(x) {
    assert x[i] >= 0;
    assert ρ[i] >= 0.0;
    m <- m + x[i];
    w <- w + x[i]*log(ρ[i]) - lgamma(x[i] + 1.0);
  }
  if (m == n) {
    return w;
  } else {
    return -inf;
  }
}

/**
 * Observe a Dirichlet variate.
 *
 * - x: The variate.
 * - α: Concentrations.
 *
 * Returns: the log probability density.
 */
function logpdf_dirichlet(x:Real[_], α:Real[_]) -> Real {
  assert length(x) == length(α);

  D:Integer <- length(x);
  w:Real <- 0.0;
  for i in 1..D {
    if x[i] < 0.0 {
      return -inf;
    }
    w <- w + (α[i] - 1.0)*log(x[i]) - lgamma(α[i]);
  }
  w <- w + lgamma(sum(α)); 
  return w;
}

/**
 * Observe a uniform variate.
 *
 * - x: The variate.
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 *
 * Returns: the log probability density.
 */
function logpdf_uniform(x:Real, l:Real, u:Real) -> Real {
  assert l <= u;

  if (x >= l && x <= u) {
    return -log(u - l);
  } else {
    return -inf;
  }
}

/**
 * Observe an exponential variate.
 *
 * - x: The variate.
 * - λ: Rate.
 *
 * Returns: the log probability density.
 */
function logpdf_exponential(x:Real, λ:Real) -> Real {
  assert 0.0 < λ;

  if (x >= 0.0) {
    return log(λ) - λ*x;
  } else {
    return -inf;
  }
}

/**
 * Observe a Weibull variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - λ: Scale.
 *
 * Returns: the log probability density.
 */
function logpdf_weibull(x:Real, k:Real, λ:Real) -> Real {
  assert 0.0 < λ;

  if (x >= 0.0) {
    return log(k) + (k - 1.0)*log(x) - k*log(λ) - pow(x/λ, k);
  } else {
    return -inf;
  }
}

/**
 * Observe a Gaussian variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Returns: the log probability density.
 */
function logpdf_gaussian(x:Real, μ:Real, σ2:Real) -> Real {
  assert 0.0 <= σ2;
  
  if (σ2 == 0.0) {
    if (x == μ) {
      return inf;
    } else {
      return -inf;
    }
  } else {
    return -0.5*(pow(x - μ, 2.0)/σ2 + log(2.0*π*σ2));
  }
}

/**
 * Observe a Student's $t$ variate.
 *
 * - x: The variate.
 * - k: Degrees of freedom.
 *
 * Returns: the log probability density.
 */
function logpdf_student_t(x:Real, k:Real) -> Real {
  assert 0.0 < k;
  auto a <- 0.5*(k + 1.0);
  return lgamma(a) - lgamma(0.5*k) - 0.5*log(π*k) - a*log1p(x*x/k);
}

/**
 * Observe a Student's $t$ variate with location and scale.
 *
 * - x: The variate.
 * - k: Degrees of freedom.
 * - μ: Location.
 * - σ2: Squared scale.
 *
 * Returns: the log probability density.
 */
function logpdf_student_t(x:Real, k:Real, μ:Real, σ2:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < σ2;
  return logpdf_student_t((x - μ)/sqrt(σ2), k) - 0.5*log(σ2);
}

/**
 * Observe a beta variate.
 *
 * - x: The variate.
 * - α: Shape.
 * - β: Shape.
 *
 * Returns: the log probability density.
 */
function logpdf_beta(x:Real, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;

  if (0.0 < x && x < 1.0) {
    return (α - 1.0)*log(x) + (β - 1.0)*log1p(-x) - lbeta(α, β);
  } else {
    return -inf;
  }
}

/**
 * Observe a $\chi^2$ variate.
 *
 * - x: The variate.
 * - ν: Degrees of freedom.
 *
 * Return: the log probability density.
 */
function logpdf_chi_squared(x:Real, ν:Real) -> Real {
  assert 0.0 < ν;
  if x > 0.0 || (x >= 0.0 && ν > 1.0) {
    auto k <- 0.5*ν;
    return (k - 1.0)*log(x) - 0.5*x - lgamma(k) - k*log(2.0);
  } else {
    return -inf;
  }
}

/**
 * Observe a gamma variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - θ: Scale.
 *
 * Returns: the log probability density.
 */
function logpdf_gamma(x:Real, k:Real, θ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < θ;
  
  if (x > 0.0) {
    return (k - 1.0)*log(x) - x/θ - lgamma(k) - k*log(θ);
  } else {
    return -inf;
  }
}

/**
 * Observe a Wishart variate.
 *
 * - X: The variate.
 * - Ψ: Scale.
 * - ν: Degrees of freedom.
 *
 * Returns: the log probability density.
 */
function logpdf_wishart(X:Real[_,_], Ψ:Real[_,_], ν:Real) -> Real {
  assert ν > rows(Ψ) - 1;
  auto p <- rows(Ψ);
  auto C <- llt(Ψ);

  return 0.5*(ν - p - 1.0)*ldet(llt(X)) - 0.5*trace(inv(C)*X) -
      0.5*ν*p*log(2.0) - 0.5*ν*ldet(C) - lgamma(0.5*ν, p);
}

/**
 * Observe an inverse-gamma variate.
 *
 * - x: The variate.
 * - α: Shape.
 * - β: Scale.
 *
 * Returns: the log probability density.
 */
function logpdf_inverse_gamma(x:Real, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;
  
  if (x > 0.0) {
    return α*log(β) - (α + 1.0)*log(x) - β/x - lgamma(α);
  } else {
    return -inf;
  }
}

/**
 * Observe an inverse Wishart variate.
 *
 * - X: The variate.
 * - Ψ: Scale.
 * - ν: Degrees of freedom.
 *
 * Returns: the log probability density.
 */
function logpdf_inverse_wishart(X:Real[_,_], Ψ:Real[_,_], ν:Real) -> Real {
  assert ν > rows(Ψ) - 1;
  auto p <- rows(Ψ);
  auto C <- llt(X);

  return 0.5*ν*ldet(llt(Ψ)) - 0.5*(ν + p - 1.0)*ldet(C) - 0.5*trace(Ψ*inv(C)) -
      0.5*ν*p*log(2.0) - lgamma(0.5*ν, p);
}

/**
 * Observe a compound-gamma variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - α: Prior shape.
 * - β: Prior scale.
 *
 * Return: the log probability density.
 */
function logpdf_inverse_gamma_gamma(x:Real, k:Real, α:Real, β:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < α;
  assert 0.0 < β;

  if x > 0.0 {
    return (k - 1)*log(x) + α*log(β) - (α + k)*log(β + x) - lbeta(α, k);
  } else {
    return -inf;
  }
}

/**
 * Observe a normal inverse-gamma variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - a2: Variance scale.
 * - α: Shape of inverse-gamma on variance.
 * - β: Scale of inverse-gamma on variance.
 *
 * Returns: the log probability density.
 */
function logpdf_normal_inverse_gamma(x:Real, μ:Real, a2:Real, α:Real,
    β:Real) -> Real {
  return logpdf_student_t(x, 2.0*α, μ, a2*β/α);
}

/**
 * Observe a beta-bernoulli variate.
 *
 * - x: The variate.
 * - α: Shape.
 * - β: Shape.
 *
 * Returns: the log probability mass.
 */
function logpdf_beta_bernoulli(x:Boolean, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;

  if (x) {
    return lbeta(1.0 + α, β) - lbeta(α, β);
  } else {
    return lbeta(α, 1.0 + β) - lbeta(α, β);
  }
}

/**
 * Observe a beta-binomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - α: Shape.
 * - β: Shape.
 *
 * Returns: the log probability mass.
 */
function logpdf_beta_binomial(x:Integer, n:Integer, α:Real, β:Real) -> Real {
  assert 0 <= n;
  assert 0.0 < α;
  assert 0.0 < β;

  if (0 <= x && x <= n) {
    return lbeta(x + α, n - x + β) - lbeta(α, β) + lchoose(n, x);
  } else {
    return -inf;
  }
}

/**
 * Observe a beta-negative-binomial variate
 *
 * - x: The variate.
 * - k: Number of successes.
 * - α: Shape.
 * - β: Shape.
 *
 * Returns: the log probability mass.
 */
function logpdf_beta_negative_binomial(x:Integer, k:Integer, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;

  if (x >= 0) {
    return lbeta(α + k, β + x) - lbeta(α, β) + lchoose(x + k - 1, x);
  } else {
    return -inf;
  }
}

/**
 * Observe a gamma-Poisson variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - θ: Scale.
 *
 * Returns: the log probability mass.
 */
function logpdf_gamma_poisson(x:Integer, k:Real, θ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < θ;
  assert k == floor(k);

  return logpdf_negative_binomial(x, Integer(k), 1.0/(θ + 1.0));
}

/**
 * Observe of a Lomax variate.
 *
 * - x: The variate.
 * - λ: Scale.
 * - α: Shape.
 *
 * Return: the log probability density.
 */
function logpdf_lomax(x:Real, λ:Real, α:Real) -> Real {
  assert 0.0 < λ;
  assert 0.0 < α;
  if x >= 0.0 {
    return log(α) - log(λ) - (α + 1.0)*log1p(x/λ);
  } else {
    return -inf;
  }
}

/**
 * Observe a Dirichlet-categorical variate.
 *
 * - x: The variate.
 * - α: Concentrations.
 *
 * Returns: the log probability mass.
 */
function logpdf_dirichlet_categorical(x:Integer, α:Real[_]) -> Real {
  if (1 <= x && x <= length(α)) {
    A:Real <- sum(α);
    return lgamma(1.0 + α[x]) - lgamma(α[x]) + lgamma(A) - lgamma(1.0 + A);
  } else {
    return -inf;
  }
}

/**
 * Observe a Dirichlet-multinomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - α: Concentrations.
 *
 * Returns: the log probability mass.
 */
function logpdf_dirichlet_multinomial(x:Integer[_], n:Integer, α:Real[_]) -> Real {
  assert length(x) == length(α);

  A:Real <- sum(α);
  m:Integer <- 0;
  w:Real <- lgamma(n + 1.0) + lgamma(A) - lgamma(n + A);
  for i in 1..length(α) {
    assert x[i] >= 0;
    m <- m + x[i];
    w <- w + lgamma(x[i] + α[i]) - lgamma(x[i] + 1.0) - lgamma(α[i]);
  }
  if (m == n) {
    return w;
  } else {
    return -inf;
  }
}

/**
 * Observe a categorical variate with Chinese restaurant process
 * prior.
 *
 * - x: The variate.
 * - α: Concentration.
 * - θ: Discount.
 * - n: Enumerated items.
 * - N: Total number of items.
 */
function logpdf_crp_categorical(k:Integer, α:Real, θ:Real,
    n:Integer[_], N:Integer) -> Real {
  K:Integer <- length(n);
  if (k > K + 1) {
    return -inf;
  } else if (k == K + 1) {
    return log(K*α + θ) - log(N + θ);
  } else {
    return log(n[k] - α) - log(N + θ);
  }
}

/**
 * Observe a Gaussian variate with a normal inverse-gamma prior.
 *
 * - x: The variate.
 * - μ: Mean.
 * - a2: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Returns: the log probability density.
 */
function logpdf_normal_inverse_gamma_gaussian(x:Real, μ:Real, a2:Real,
    α:Real, β:Real) -> Real {
  return logpdf_student_t(x, 2.0*α, μ, (β/α)*(1.0 + a2));
}

/**
 * Observe a Gaussian variate with a normal inverse-gamma prior with linear
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
 * Returns: the log probability density.
 */
function logpdf_linear_normal_inverse_gamma_gaussian(x:Real, a:Real,
    μ:Real, a2:Real, c:Real, α:Real, β:Real) -> Real {
  return logpdf_student_t(x, 2.0*α, a*μ + c, (β/α)*(1.0 + a*a*a2));
}

/**
 * Observe a multivariate Gaussian variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - Σ: Covariance.
 *
 * Returns: the log probability density.
 */
function logpdf_multivariate_gaussian(x:Real[_], μ:Real[_], Σ:Real[_,_]) ->
    Real {
  auto D <- length(μ);
  auto C <- llt(Σ);
  return -0.5*(dot(x - μ, solve(C, x - μ)) + D*log(2.0*π) + ldet(C));
}

/**
 * Observe a multivariate Gaussian distribution with independent elements.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Returns: the log probability density.
 */
function logpdf_multivariate_gaussian(x:Real[_], μ:Real[_], σ2:Real[_]) -> Real {
  auto D <- length(μ);
  auto w <- 0.0;
  for d in 1..D {
    w <- w + logpdf_gaussian(x[d], μ[d], σ2[d]);
  }
  return w;
}

/**
 * Observe a multivariate Gaussian distribution with independent elements of
 * identical variance.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Returns: the log probability density.
 */
function logpdf_multivariate_gaussian(x:Real[_], μ:Real[_], σ2:Real) -> Real {
  auto D <- length(μ);
  return -0.5*(dot(x - μ)/σ2 + D*log(2.0*π*σ2));
}

/**
 * Observe a multivariate normal inverse-gamma variate.
 *
 * - x: The variate.
 * - ν: Precision times mean.
 * - Λ: Precision.
 * - α: Shape of inverse-gamma on scale.
 * - β: Scale of inverse-gamma on scale.
 *
 * Returns: the log probability density.
 */
function logpdf_multivariate_normal_inverse_gamma(x:Real[_], ν:Real[_],
    Λ:LLT, α:Real, β:Real) -> Real {
  return logpdf_multivariate_student_t(x, 2.0*α, solve(Λ, ν), (β/α)*inv(Λ));
}

/**
 * Observe a multivariate Gaussian variate with a multivariate normal
 * inverse-gamma prior.
 *
 * - x: The variate.
 * - ν: Precision times mean.
 * - Λ: Precision.
 * - α: Shape of the inverse-gamma.
 * - γ: Scale accumulator of the inverse-gamma.
 *
 * Returns: the log probability density.
 */
function logpdf_multivariate_normal_inverse_gamma_multivariate_gaussian(x:Real[_],
    ν:Real[_], Λ:LLT, α:Real, γ:Real) -> Real {
  auto D <- length(ν);
  auto β <- γ - 0.5*dot(solve(cholesky(Λ), ν));
  return logpdf_multivariate_student_t(x, 2.0*α, solve(Λ, ν), (β/α)*(identity(D) + inv(Λ)));
}

/**
 * Observe a multivariate Gaussian variate with a multivariate linear normal
 * inverse-gamma prior with linear transformation.
 *
 * - x: The variate.
 * - A: Scale.
 * - ν: Precision times mean.
 * - Λ: Precision.
 * - c: Offset.
 * - α: Shape of the inverse-gamma.
 * - γ: Scale accumulator of the inverse-gamma.
 *
 * Returns: the log probability density.
 */
function logpdf_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(x:Real[_],
    A:Real[_,_], ν:Real[_], Λ:LLT, c:Real[_], α:Real, γ:Real) -> Real {
  auto β <- γ - 0.5*dot(solve(cholesky(Λ), ν));
  return logpdf_multivariate_student_t(x, 2.0*α, A*solve(Λ, ν) + c,
      (β/α)*(identity(rows(A)) + A*solve(Λ, transpose(A))));
}

/**
 * Observe a matrix Gaussian distribution.
 *
 * - X: The variate.
 * - M: Mean.
 * - U: Among-row covariance.
 * - V: Among-column covariance.
 *
 * Returns: the log probability density.
 */
function logpdf_matrix_gaussian(X:Real[_,_], M:Real[_,_], U:Real[_,_],
    V:Real[_,_]) -> Real {
  auto n <- rows(M);
  auto p <- columns(M);
  auto C <- llt(U);
  auto D <- llt(V);
  return -0.5*(trace(inv(D)*transpose(X - M)*inv(C)*(X - M)) +
      n*p*log(2.0*π) + n*ldet(D) + p*ldet(C));
}

/**
 * Observe a matrix Gaussian distribution with independent columns.
 *
 * - X: The variate.
 * - M: Mean.
 * - U: Among-row covariance.
 * - σ2: Among-column variances.
 *
 * Returns: the log probability density.
 */
function logpdf_matrix_gaussian(X:Real[_,_], M:Real[_,_], U:Real[_,_],
    σ2:Real[_]) -> Real {
  auto n <- rows(M);
  auto p <- columns(M);
  auto C <- llt(U);
  return -0.5*(trace(inv(diagonal(σ2))*transpose(X - M)*inv(C)*(X - M)) +
      n*p*log(2.0*π) + n*log_sum(σ2) + p*ldet(C));
}

/**
 * Observe a matrix Gaussian distribution with independent rows.
 *
 * - X: The variate.
 * - M: Mean.
 * - V: Among-column covariance.
 *
 * Returns: the log probability density.
 */
function logpdf_matrix_gaussian(X:Real[_,_], M:Real[_,_], V:Real[_,_]) ->
    Real {
  auto n <- rows(M);
  auto p <- columns(M);
  auto D <- llt(V);
  return -0.5*(trace(inv(D)*transpose(X - M)*(X - M)) + n*p*log(2.0*π) +
      n*ldet(D));
}

/**
 * Observe a matrix Gaussian distribution with independent elements.
 *
 * - X: The variate.
 * - M: Mean.
 * - σ2: Among-column variances.
 *
 * Returns: the log probability density.
 */
function logpdf_matrix_gaussian(X:Real[_,_], M:Real[_,_], σ2:Real[_]) ->
    Real {
  auto n <- rows(M);
  auto p <- columns(M);
  return -0.5*(trace(inv(diagonal(σ2))*transpose(X - M)*(X - M)) +
      n*p*log(2.0*π) + n*log_sum(σ2));
}

/**
 * Observe a matrix Gaussian distribution with independent elements
 * of identical variance.
 *
 * - X: The variate.
 * - M: Mean.
 * - σ2: Variance.
 *
 * Returns: the log probability density.
 */
function logpdf_matrix_gaussian(X:Real[_,_], M:Real[_,_], σ2:Real) ->
    Real {
  auto n <- rows(M);
  auto p <- columns(M);
  return -0.5*(trace(transpose(X - M)*(X - M)/σ2) +
      n*p*log(2.0*π*σ2));
}

/**
 * Observe a matrix normal-inverse-gamma variate.
 *
 * - X: The variate.
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - α: Variance shape.
 * - β: Variance scales.
 *
 * Returns: the log probability density.
 */
function logpdf_matrix_normal_inverse_gamma(X:Real[_,_], N:Real[_,_], Λ:LLT,
    α:Real, β:Real[_]) -> Real {
  auto M <- solve(Λ, N);
  auto Σ <- inv(Λ);
  return logpdf_matrix_student_t(X, 2.0*α, M, Σ, β/α);
}

/**
 * Observe a Gaussian variate with matrix normal inverse-gamma prior.
 *
 * - X: The variate.
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - α: Variance shape.
 * - γ: Variance scale accumulators.
 *
 * Returns: the log probability density.
 */
function logpdf_matrix_normal_inverse_gamma_matrix_gaussian(X:Real[_,_],
    N:Real[_,_], Λ:LLT, α:Real, γ:Real[_]) -> Real {
  M:Real[_,_] <- solve(Λ, N);
  auto Σ <- identity(rows(M)) + inv(Λ);
  auto β <- γ - 0.5*diagonal(transpose(M)*N);
  return logpdf_matrix_student_t(X, 2.0*α, M, Σ, β/α);
}

/**
 * Observe a Gaussian variate with matrix normal inverse-gamma prior.
 *
 * - X: The variate.
 * - A: Scale.
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - C: Offset.
 * - α: Variance shape.
 * - γ: Variance scale accumulators.
 *
 * Returns: the log probability density.
 */
function logpdf_linear_matrix_normal_inverse_gamma_matrix_gaussian(
    X:Real[_,_], A:Real[_,_], N:Real[_,_], Λ:LLT, C:Real[_,_], α:Real,
    γ:Real[_]) -> Real {
  M:Real[_,_] <- solve(Λ, N);
  auto β <- γ - 0.5*diagonal(transpose(M)*N);
  auto Σ <- identity(rows(A)) + A*solve(Λ, transpose(A));
  return logpdf_matrix_student_t(X, 2.0*α, A*M + C, Σ, β/α);
}

/**
 * Observe a matrix normal-inverse-Wishart variate.
 *
 * - X: The variate.
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - Ψ: Prior variance shape.
 * - k: Prior degrees of freedom.
 *
 * Returns: the log probability density.
 */
function logpdf_matrix_normal_inverse_wishart(X:Real[_,_], N:Real[_,_],
    Λ:LLT,  Ψ:Real[_,_], k:Real) -> Real {
  auto p <- columns(N);
  auto M <- solve(Λ, N);
  auto Σ <- inv(Λ)/(k - p + 1.0);
  return logpdf_matrix_student_t(X, k - p + 1.0, M, Σ, Ψ);
}

/**
 * Observe a Gaussian variate with matrix-normal-inverse-Wishart prior.
 *
 * - X: The variate.
 * - N: Prior precision times mean matrix.
 * - Λ: Prior precision.
 * - Ψ: Prior variance shape.
 * - k: Prior degrees of freedom.
 *
 * Returns: the log probability density.
 */
function logpdf_matrix_normal_inverse_wishart_matrix_gaussian(X:Real[_,_],
    N:Real[_,_], Λ:LLT, Ψ:Real[_,_], k:Real) -> Real {
  auto p <- columns(N);
  auto M <- solve(Λ, N);
  auto Σ <- (identity(rows(N)) + inv(Λ))/(k - p + 1.0);
  return logpdf_matrix_student_t(X, k - p + 1.0, M, Σ, Ψ);
}

/**
 * Observe a Gaussian variate with linear transformation of a
 * matrix-normal-inverse-Wishart prior.
 *
 * - X: The variate.
 * - A: Scale.
 * - N: Prior precision times mean matrix.
 * - Λ: Prior precision.
 * - C: Offset.
 * - Ψ: Prior variance shape.
 * - k: Prior degrees of freedom.
 *
 * Returns: the log probability density.
 */
function logpdf_linear_matrix_normal_inverse_wishart_matrix_gaussian(
    X:Real[_,_], A:Real[_,_], N:Real[_,_], Λ:LLT, C:Real[_,_], Ψ:Real[_,_],
    k:Real) -> Real {
  auto p <- columns(N);
  auto M <- solve(Λ, N);
  auto Σ <- (identity(rows(A)) + A*solve(Λ, transpose(A)))/(k - p + 1.0);
  return logpdf_matrix_student_t(X, k - p + 1.0, A*M + C, Σ, Ψ);
}

/**
 * Observe a multivariate Student's $t$-distribution variate with location
 * and scale.
 *
 * - x: The variate.
 * - k: Degrees of freedom.
 * - m: Mean.
 * - Σ: Covariance.
 *
 * Returns: the log probability density.
 */
function logpdf_multivariate_student_t(x:Real[_], k:Real, μ:Real[_], Σ:Real[_,_])
    -> Real {
  auto n <- length(μ);
  auto a <- 0.5*(k + n);
  auto C <- llt(Σ);
  return lgamma(a) - lgamma(0.5*k) - 0.5*n*log(k*π) - 0.5*ldet(C) -
      a*log1p(dot(x - μ, solve(C, x - μ))/k) ;
}

/**
 * Observe a multivariate Student's $t$-distribution variate with location
 * and diagonal scale.
 *
 * - x: The variate.
 * - k: Degrees of freedom.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Returns: the log probability density.
 */
function logpdf_multivariate_student_t(x:Real[_], k:Real, μ:Real[_],
    σ2:Real) -> Real {
  auto n <- length(μ);
  auto a <- 0.5*(k + n);
  return lgamma(a) - lgamma(0.5*k) - 0.5*n*log(k*π) - 0.5*n*log(σ2) -
      a*log1p(dot(x - μ)/(σ2*k));
}

/**
 * Observe a matrix Student's $t$-distribution variate with location
 * and scale.
 *
 * - X: The variate.
 * - k: Degrees of freedom.
 * - M: Mean.
 * - U: Among-row covariance.
 * - V: Among-column covariance.
 *
 * Returns: the log probability density.
 */
function logpdf_matrix_student_t(X:Real[_,_], k:Real, M:Real[_,_],
    U:Real[_,_], V:Real[_,_]) -> Real {
  auto n <- rows(M);
  auto p <- columns(M);
  auto a <- 0.5*(k + n + p - 1.0);
  auto C <- llt(U);
  auto D <- llt(V);
  auto E <- identity(n) + solve(C, X - M)*solve(D, transpose(X - M));
  
  return lgamma(a, p) - lgamma(0.5*(k + p - 1.0), p) - 0.5*n*p*log(π) -
      0.5*n*ldet(C) - 0.5*p*ldet(D) - a*ldet(E);
}

/**
 * Observe a matrix Student's $t$-distribution variate with location
 * and diagonal scale.
 *
 * - X: The variate.
 * - k: Degrees of freedom.
 * - M: Mean.
 * - U: Among-row covariance.
 * - v: Independent within-column covariance.
 *
 * Returns: the log probability density.
 */
function logpdf_matrix_student_t(X:Real[_,_], k:Real, M:Real[_,_],
    U:Real[_,_], v:Real[_]) -> Real {
  auto n <- rows(M);
  auto p <- columns(M);
  auto a <- 0.5*(k + n + p - 1.0);
  auto C <- llt(U);
  auto E <- identity(n) + solve(C, X - M)*solve(diagonal(v), transpose(X - M));
  
  return lgamma(a, p) - lgamma(0.5*(k + p - 1.0), p) - 0.5*n*p*log(π) -
      0.5*n*ldet(C) - 0.5*p*log_sum(v) - a*ldet(E);
}

/**
 * Observe a multivariate uniform distribution.
 *
 * - x: The variate.
 * - l: Lower bound of hyperrectangle.
 * - u: Upper bound of hyperrectangle.
 *
 * Returns: the log probability density.
 */
function logpdf_independent_uniform(x:Real[_], l:Real[_], u:Real[_]) -> Real {
  assert length(x) > 0;
  assert length(l) == length(x);
  assert length(u) == length(x);

  D:Integer <- length(l);
  w:Real <- 0.0;
  for d in 1..D {
    w <- w + logpdf_uniform(x[d], l[d], u[d]);
  }
  return w;
}

/**
 * Observe a multivariate uniform distribution over integers.
 *
 * - x: The variate.
 * - l: Lower bound of hyperrectangle.
 * - u: Upper bound of hyperrectangle.
 *
 * Returns: the log probability mass.
 */
function logpdf_independent_uniform_int(x:Integer[_], l:Integer[_], u:Integer[_]) -> Real {
  assert length(x) > 0;
  assert length(l) == length(x);
  assert length(u) == length(x);
  
  D:Integer <- length(x);
  w:Real <- 0.0;
  for d in 1..D {
    w <- w + logpdf_uniform_int(x[d], l[d], u[d]);
  }
  return w;
}
