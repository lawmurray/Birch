/**
 * Observe a Bernoulli variate.
 *
 * - x: The variate.
 * - ρ: Probability of a true result.
 *
 * Returns: the log probability mass.
 */
function lazy_bernoulli(x:Boolean, ρ:Expression<Real>) -> Expression<Real> {
  assert 0.0 <= ρ && ρ <= 1.0;
  if (x) {
    return log(ρ);
  } else {
    return Boxed(log1p(-ρ));
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
function lazy_delta(x:Integer, μ:Expression<Integer>) -> Expression<Real> {
  if (x == μ) {
    return Boxed(0.0);
  } else {
    return Boxed(-inf);
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
function lazy_binomial(x:Integer, n:Expression<Integer>, ρ:Expression<Real>) -> Expression<Real> {
  assert 0 <= n;
  assert 0.0 <= ρ && ρ <= 1.0;

  if ρ == 0.0 || ρ == 1.0 {
    if x == n*ρ {
      return Boxed(0.0);
    } else {
      return Boxed(-inf);
    }
  } else if 0 <= x && x <= n {
    return x*log(ρ) + (n - x)*log1p(-ρ) + lchoose(n, x);
  } else {
    return Boxed(-inf);
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
function lazy_negative_binomial(x:Integer, k:Expression<Integer>, ρ:Expression<Real>) -> Expression<Real> {
  assert 0 < k;
  assert 0.0 <= ρ && ρ <= 1.0;

  if (x >= 0) {
    return k*log(ρ) + x*log1p(-ρ) + lchoose(x + k - 1, x);
  } else {
    return Boxed(-inf);
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
function lazy_poisson(x:Expression<Integer>, λ:Expression<Real>) -> Expression<Real> {
  return x*log(λ) - λ - lgamma(x + 1);
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
function lazy_uniform(x:Expression<Real>, l:Expression<Real>, u:Expression<Real>) -> Expression<Real> {
  assert l <= u;

  if (x >= l && x <= u) {
    return -log(u - l);
  } else {
    return Boxed(-inf);
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
function lazy_exponential(x:Expression<Real>, λ:Expression<Real>) -> Expression<Real> {
  assert 0.0 < λ;

  if (x >= 0.0) {
    return log(λ) - λ*x;
  } else {
    return Boxed(-inf);
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
function lazy_weibull(x:Expression<Real>, k:Expression<Real>, λ:Expression<Real>) -> Expression<Real> {
  assert 0.0 < λ;

  if (x >= 0.0) {
    return log(k) + (k - 1.0)*log(x) - k*log(λ) - pow(x/λ, k);
  } else {
    return Boxed(-inf);
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
function lazy_gaussian(x:Expression<Real>, μ:Expression<Real>, σ2:Expression<Real>) -> Expression<Real> {
  return -0.5*(pow(x - μ, 2.0)/σ2 + log(2.0*π*σ2));
}

/**
 * Observe a Student's $t$ variate.
 *
 * - x: The variate.
 * - k: Degrees of freedom.
 *
 * Returns: the log probability density.
 */
function lazy_student_t(x:Expression<Real>, k:Expression<Real>) -> Expression<Real> {
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
function lazy_student_t(x:Expression<Real>, k:Expression<Real>, μ:Expression<Real>, σ2:Expression<Real>) -> Expression<Real> {
  assert 0.0 < k;
  assert 0.0 < σ2;
  return lazy_student_t((x - μ)/sqrt(σ2), k) - 0.5*log(σ2);
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
function lazy_beta(x:Expression<Real>, α:Expression<Real>, β:Expression<Real>) -> Expression<Real> {
  assert 0.0 < α;
  assert 0.0 < β;

  if (0.0 < x && x < 1.0) {
    return (α - 1.0)*log(x) + (β - 1.0)*log1p(-x) - lbeta(α, β);
  } else {
    return Boxed(-inf);
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
function lazy_chi_squared(x:Expression<Real>, ν:Expression<Real>) -> Expression<Real> {
  assert 0.0 < ν;
  if x > 0.0 || (x >= 0.0 && ν > 1.0) {
    auto k <- 0.5*ν;
    return (k - 1.0)*log(x) - 0.5*x - lgamma(k) - k*log(2.0);
  } else {
    return Boxed(-inf);
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
function lazy_gamma(x:Expression<Real>, k:Expression<Real>, θ:Expression<Real>) -> Expression<Real> {
  assert 0.0 < k;
  assert 0.0 < θ;
  
  if (x > 0.0) {
    return (k - 1.0)*log(x) - x/θ - lgamma(k) - k*log(θ);
  } else {
    return Boxed(-inf);
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
function lazy_wishart(X:Expression<Real[_,_]>, Ψ:Expression<Real[_,_]>, ν:Expression<Real>) -> Expression<Real> {
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
function lazy_inverse_gamma(x:Expression<Real>, α:Expression<Real>, β:Expression<Real>) -> Expression<Real> {
  assert 0.0 < α;
  assert 0.0 < β;
  
  if (x > 0.0) {
    return α*log(β) - (α + 1.0)*log(x) - β/x - lgamma(α);
  } else {
    return Boxed(-inf);
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
function lazy_inverse_wishart(X:Expression<Real[_,_]>, Ψ:Expression<Real[_,_]>, ν:Expression<Real>) -> Expression<Real> {
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
function lazy_inverse_gamma_gamma(x:Expression<Real>, k:Expression<Real>, α:Expression<Real>, β:Expression<Real>) -> Expression<Real> {
  assert 0.0 < k;
  assert 0.0 < α;
  assert 0.0 < β;

  if x > 0.0 {
    return (k - 1)*log(x) + α*log(β) - (α + k)*log(β + x) - lbeta(α, k);
  } else {
    return Boxed(-inf);
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
function lazy_normal_inverse_gamma(x:Expression<Real>, μ:Expression<Real>, a2:Expression<Real>, α:Expression<Real>,
    β:Expression<Real>) -> Expression<Real> {
  return lazy_student_t(x, 2.0*α, μ, a2*β/α);
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
function lazy_beta_binomial(x:Integer, n:Expression<Integer>,
    α:Expression<Real>, β:Expression<Real>) -> Expression<Real> {
  assert 0 <= n;
  assert 0.0 < α;
  assert 0.0 < β;

  if (0 <= x && x <= n) {
    return lbeta(x + α, n - x + β) - lbeta(α, β) + lchoose(n, x);
  } else {
    return Boxed(-inf);
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
function lazy_beta_negative_binomial(x:Integer, k:Expression<Integer>, α:Expression<Real>, β:Expression<Real>) -> Expression<Real> {
  assert 0.0 < α;
  assert 0.0 < β;

  if (x >= 0) {
    return lbeta(α + k, β + x) - lbeta(α, β) + lchoose(x + k - 1, x);
  } else {
    return Boxed(-inf);
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
function lazy_gamma_poisson(x:Integer, k:Expression<Integer>, θ:Expression<Real>) -> Expression<Real> {
  assert 0.0 < k;
  assert 0.0 < θ;
  assert k == floor(k);

  return lazy_negative_binomial(x, k, 1.0/(θ + 1.0));
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
function lazy_lomax(x:Expression<Real>, λ:Expression<Real>, α:Expression<Real>) -> Expression<Real> {
  assert 0.0 < λ;
  assert 0.0 < α;
  if x >= 0.0 {
    return log(α) - log(λ) - (α + 1.0)*log1p(x/λ);
  } else {
    return Boxed(-inf);
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
function lazy_normal_inverse_gamma_gaussian(x:Expression<Real>, μ:Expression<Real>, a2:Expression<Real>,
    α:Expression<Real>, β:Expression<Real>) -> Expression<Real> {
  return lazy_student_t(x, 2.0*α, μ, (β/α)*(1.0 + a2));
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
function lazy_linear_normal_inverse_gamma_gaussian(x:Expression<Real>, a:Expression<Real>,
    μ:Expression<Real>, a2:Expression<Real>, c:Expression<Real>, α:Expression<Real>, β:Expression<Real>) -> Expression<Real> {
  return lazy_student_t(x, 2.0*α, a*μ + c, (β/α)*(1.0 + a*a*a2));
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
function lazy_multivariate_gaussian(x:Expression<Real[_]>, μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) ->
    Real {
  auto D <- length(μ);
  auto C <- llt(Σ);
  return -0.5*(dot(x - μ, solve(C, x - μ)) + D*log(2.0*π) + ldet(C));
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
function lazy_multivariate_gaussian(x:Expression<Real[_]>, μ:Expression<Real[_]>, σ2:Expression<Real>) -> Expression<Real> {
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
function lazy_multivariate_normal_inverse_gamma(x:Expression<Real[_]>, ν:Expression<Real[_]>,
    Λ:Expression<Real[_,_]>, α:Expression<Real>, β:Expression<Real>) -> Expression<Real> {
  ///@todo return lazy_multivariate_student_t(x, 2.0*α, solve(Λ, ν), (β/α)*inv(Λ));
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
function lazy_multivariate_normal_inverse_gamma_multivariate_gaussian(x:Expression<Real[_]>,
    ν:Expression<Real[_]>, Λ:Expression<Real[_,_]>, α:Expression<Real>, γ:Expression<Real>) -> Expression<Real> {
  auto D <- length(ν);
  auto β <- γ - 0.5*dot(solve(cholesky(Λ), ν));
  ///@todo return lazy_multivariate_student_t(x, 2.0*α, solve(Λ, ν), (β/α)*(identity(D) + inv(Λ)));
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
function lazy_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(x:Expression<Real[_]>,
    A:Expression<Real[_,_]>, ν:Expression<Real[_]>, Λ:Expression<Real[_,_]>, c:Expression<Real[_]>, α:Expression<Real>, γ:Expression<Real>) -> Expression<Real> {
  auto β <- γ - 0.5*dot(solve(cholesky(Λ), ν));
  ///@todo
  //return lazy_multivariate_student_t(x, 2.0*α, A*solve(Λ, ν) + c,
  //    (β/α)*(identity(rows(A)) + A*solve(Λ, transpose(A))));
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
function lazy_matrix_gaussian(X:Expression<Real[_,_]>, M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) -> Expression<Real> {
  auto n <- rows(M);
  auto p <- columns(M);
  auto C <- llt(U);
  auto D <- llt(V);
  ///@todo
  //return -0.5*(trace(inv(D)*transpose(X - M)*inv(C)*(X - M)) +
  //    n*p*log(2.0*π) + n*ldet(D) + p*ldet(C));
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
function lazy_matrix_gaussian(X:Expression<Real[_,_]>, M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    σ2:Expression<Real[_]>) -> Expression<Real> {
  auto n <- rows(M);
  auto p <- columns(M);
  auto C <- llt(U);
  ///@todo
  //return -0.5*(trace(inv(diagonal(σ2))*transpose(X - M)*inv(C)*(X - M)) +
  //    n*p*log(2.0*π) + n*log_sum(σ2) + p*ldet(C));
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
function lazy_matrix_gaussian(X:Expression<Real[_,_]>, M:Expression<Real[_,_]>, V:Expression<Real[_,_]>) ->
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
function lazy_matrix_gaussian(X:Expression<Real[_,_]>, M:Expression<Real[_,_]>, σ2:Expression<Real[_]>) ->
    Real {
  auto n <- rows(M);
  auto p <- columns(M);
  return -0.5*(trace(inv(diagonal(σ2))*transpose(X - M)*(X - M)) +
      n*p*log(2.0*π) + n*log_sum(σ2));
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
function lazy_matrix_normal_inverse_gamma(X:Expression<Real[_,_]>, N:Expression<Real[_,_]>, Λ:Expression<Real[_,_]>,
    α:Expression<Real>, β:Expression<Real[_]>) -> Expression<Real> {
  auto M <- solve(Λ, N);
  auto Σ <- inv(Λ);
  ///@todo
  //return lazy_matrix_student_t(X, 2.0*α, M, Σ, β/α);
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
function lazy_matrix_normal_inverse_gamma_matrix_gaussian(X:Expression<Real[_,_]>,
    N:Expression<Real[_,_]>, Λ:Expression<Real[_,_]>, α:Expression<Real>, γ:Expression<Real[_]>) -> Expression<Real> {
  auto M <- solve(Λ, N);
  auto Σ <- identity(rows(M)) + inv(Λ);
  auto β <- γ - 0.5*diagonal(transpose(M)*N);
  ///@todo
  //return lazy_matrix_student_t(X, 2.0*α, M, Σ, β/α);
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
function lazy_linear_matrix_normal_inverse_gamma_matrix_gaussian(
    X:Expression<Real[_,_]>, A:Expression<Real[_,_]>, N:Expression<Real[_,_]>, Λ:Expression<Real[_,_]>, C:Expression<Real[_,_]>, α:Expression<Real>,
    γ:Expression<Real[_]>) -> Expression<Real> {
  auto M <- solve(Λ, N);
  auto β <- γ - 0.5*diagonal(transpose(M)*N);
  auto Σ <- identity(rows(A)) + A*solve(Λ, transpose(A));
  ///@todo
  //return lazy_matrix_student_t(X, 2.0*α, A*M + C, Σ, β/α);
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
function lazy_matrix_normal_inverse_wishart(X:Expression<Real[_,_]>, N:Expression<Real[_,_]>,
    Λ:Expression<Real[_,_]>,  Ψ:Expression<Real[_,_]>, k:Expression<Real>) -> Expression<Real> {
  auto p <- columns(N);
  auto M <- solve(Λ, N);
  auto Σ <- inv(Λ)/(k - p + 1.0);
  ///@todo
  //return lazy_matrix_student_t(X, k - p + 1.0, M, Σ, Ψ);
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
function lazy_matrix_normal_inverse_wishart_matrix_gaussian(X:Expression<Real[_,_]>,
    N:Expression<Real[_,_]>, Λ:Expression<Real[_,_]>, Ψ:Expression<Real[_,_]>, k:Expression<Real>) -> Expression<Real> {
  auto p <- columns(N);
  auto M <- solve(Λ, N);
  auto Σ <- (identity(rows(N)) + inv(Λ))/(k - p + 1.0);
  ///@todo
  //return lazy_matrix_student_t(X, k - p + 1.0, M, Σ, Ψ);
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
function lazy_linear_matrix_normal_inverse_wishart_matrix_gaussian(
    X:Expression<Real[_,_]>, A:Expression<Real[_,_]>, N:Expression<Real[_,_]>, Λ:Expression<Real[_,_]>, C:Expression<Real[_,_]>, Ψ:Expression<Real[_,_]>,
    k:Expression<Real>) -> Expression<Real> {
  auto p <- columns(N);
  auto M <- solve(Λ, N);
  auto Σ <- (identity(rows(A)) + A*solve(Λ, transpose(A)))/(k - p + 1.0);
  ///@todo
  //return lazy_matrix_student_t(X, k - p + 1.0, A*M + C, Σ, Ψ);
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
function lazy_multivariate_student_t(x:Expression<Real[_]>, k:Expression<Real>, μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>)
    -> Expression<Real> {
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
function lazy_multivariate_student_t(x:Expression<Real[_]>, k:Expression<Real>, μ:Expression<Real[_]>,
    σ2:Expression<Real>) -> Expression<Real> {
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
function lazy_matrix_student_t(X:Expression<Real[_,_]>, k:Expression<Real>, M:Expression<Real[_,_]>,
    U:Expression<Real[_,_]>, V:Expression<Real[_,_]>) -> Expression<Real> {
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
function lazy_matrix_student_t(X:Expression<Real[_,_]>, k:Expression<Real>, M:Expression<Real[_,_]>,
    U:Expression<Real[_,_]>, v:Expression<Real[_]>) -> Expression<Real> {
  auto n <- rows(M);
  auto p <- columns(M);
  auto a <- 0.5*(k + n + p - 1.0);
  auto C <- llt(U);
  auto E <- identity(n) + solve(C, X - M)*solve(diagonal(v), transpose(X - M));
  
  return lgamma(a, p) - lgamma(0.5*(k + p - 1.0), p) - 0.5*n*p*log(π) -
      0.5*n*ldet(C) - 0.5*p*log_sum(v) - a*ldet(E);
}
