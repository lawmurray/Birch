cpp{{
#include <random>

thread_local static std::mt19937_64 rng;
}}

/**
 * Seed the pseudorandom number generator.
 *
 * - seed: Seed value.
 */
function seed(s:Integer) {
  cpp{{
  #pragma omp parallel num_threads(libbirch::get_max_threads())
  {
    rng.seed(s + libbirch::get_thread_num());
  }
  }}
}

/**
 * Seed the pseudorandom number generator with entropy.
 */
function seed() {
  cpp{{
  //std::random_device rd;
  #pragma omp parallel num_threads(libbirch::get_max_threads())
  {
    #pragma omp critical
    rng.seed(0);
  }
  }}
}

/**
 * Simulate a Bernoulli distribution.
 *
 * - ρ: Probability of a true result.
 */
function simulate_bernoulli(ρ:Real) -> Boolean {
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp{{
  return std::bernoulli_distribution(ρ)(rng);
  }}
}

/**
 * Simulate a delta distribution.
 *
 * - μ: Location.
 */
function simulate_delta(μ:Integer) -> Integer {
  return μ;
}

/**
 * Simulate a binomial distribution.
 *
 * - n: Number of trials.
 * - ρ: Probability of a true result.
 */
function simulate_binomial(n:Integer, ρ:Real) -> Integer {
  assert 0 <= n;
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp{{
  return std::binomial_distribution<bi::type::Integer>(n, ρ)(rng);
  }}
}

/**
 * Simulate a negative binomial distribution.
 *
 * - k: Number of successes before the experiment is stopped.
 * - ρ: Probability of success.
 *
 * Returns the number of failures.
 */
function simulate_negative_binomial(k:Integer, ρ:Real) -> Integer {
  assert 0 < k;
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp{{
  return std::negative_binomial_distribution<bi::type::Integer>(k, ρ)(rng);
  }}
}

/**
 * Simulate a Poisson distribution.
 *
 * - λ: Rate.
 */
function simulate_poisson(λ:Real) -> Integer {
  assert 0.0 <= λ;
  if (λ > 0.0) {
    cpp{{
    return std::poisson_distribution<bi::type::Integer>(λ)(rng);
    }}
  } else {
    return 0;
  }
}

/**
 * Simulate a categorical distribution.
 *
 * - ρ: Category probabilities. These should sum to one.
 */
function simulate_categorical(ρ:Real[_]) -> Integer {
  return simulate_categorical(ρ, 1.0);
}

/**
 * Simulate a categorical distribution.
 *
 * - ρ: Unnormalized category probabilities.
 * - Z: Sum of the unnormalized category probabilities.
 */
function simulate_categorical(ρ:Real[_], Z:Real) -> Integer {
  assert length(ρ) > 0;
  assert abs(sum(ρ) - Z) < 1.0e-6;

  u:Real <- simulate_uniform(0.0, Z);
  x:Integer <- 1;
  P:Real <- ρ[1];
  while (P < u) {
    assert x <= length(ρ);
    x <- x + 1;
    assert 0.0 <= ρ[x];
    P <- P + ρ[x];
    assert P < Z + 1.0e-6;
  }
  return x;
}

/**
 * Simulate a multinomial distribution.
 *
 * - n: Number of trials.
 * - ρ: Category probabilities. These should sum to one.
 *
 * This uses an $\mathcal{O}(N)$ implementation based on:
 *
 * Bentley, J. L. and J. B. Saxe (1979). Generating sorted lists of random
 * numbers. Technical Report 2450, Carnegie Mellon University, Computer
 * Science Department.
 */
function simulate_multinomial(n:Integer, ρ:Real[_]) -> Integer[_] {
  return simulate_multinomial(n, ρ, 1.0);
}

/**
 * Simulate a compound-gamma distribution.
 *
 * - k: Shape.
 * - α: Shape.
 * - β: Scale.
 *
 */
 function simulate_inverse_gamma_gamma(k:Real, α:Real, β:Real) -> Real {
    return simulate_gamma(k, simulate_inverse_gamma(α, β));
 }

/**
 * Simulate a multinomial distribution.
 *
 * - n: Number of trials.
 * - ρ: Unnormalized category probabilities.
 * - Z: Sum of the unnormalized category probabilities.
 *
 * This uses an $\mathcal{O}(N)$ implementation based on:
 *
 * Bentley, J. L. and J. B. Saxe (1979). Generating sorted lists of random
 * numbers. Technical Report 2450, Carnegie Mellon University, Computer
 * Science Department.
 */
function simulate_multinomial(n:Integer, ρ:Real[_], Z:Real) -> Integer[_] {
  assert length(ρ) > 0;
  assert abs(sum(ρ) - Z) < 1.0e-6;

  D:Integer <- length(ρ);
  R:Real <- ρ[D];
  lnMax:Real <- 0.0;
  j:Integer <- D;
  i:Integer <- n;
  u:Real;
  x:Integer[_] <- vector(0, D);
    
  while i > 0 {
    u <- simulate_uniform(0.0, 1.0);
    lnMax <- lnMax + log(u)/i;
    u <- Z*exp(lnMax);
    while u < Z - R {
      j <- j - 1;
      R <- R + ρ[j];
    }
    x[j] <- x[j] + 1;
    i <- i - 1;
  }
  while j > 1 {
    j <- j - 1;
    x[j] <- 0;
  }
  return x;
}

/**
 * Simulate a Dirichlet distribution.
 *
 * - α: Concentrations.
 */
function simulate_dirichlet(α:Real[_]) -> Real[_] {
  D:Integer <- length(α);
  x:Real[D];
  z:Real <- 0.0;

  for i in 1..D {
    x[i] <- simulate_gamma(α[i], 1.0);
    z <- z + x[i];
  }
  z <- 1.0/z;
  for i in 1..D {
    x[i] <- z*x[i];
  }  
  return x;
}

/**
 * Simulate a Dirichlet distribution.
 *
 * - α: Concentration.
 * - D: Number of dimensions.
 */
function simulate_dirichlet(α:Real, D:Integer) -> Real[_] {
  assert D > 0;
  x:Real[D];
  z:Real <- 0.0;

  for i in 1..D {
    x[i] <- simulate_gamma(α, 1.0);
    z <- z + x[i];
  }
  z <- 1.0/z;
  for i in 1..D {
    x[i] <- z*x[i];
  }
  return x;
}

/**
 * Simulate a uniform distribution.
 *
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 */
function simulate_uniform(l:Real, u:Real) -> Real {
  assert l <= u;
  cpp{{
  return std::uniform_real_distribution<bi::type::Real>(l, u)(rng);
  }}
}

/**
 * Simulate a uniform distribution on an integer range.
 *
 * - l: Lower bound of range.
 * - u: Upper bound of range.
 */
function simulate_uniform_int(l:Integer, u:Integer) -> Integer {
  assert l <= u;
  cpp{{
  return std::uniform_int_distribution<bi::type::Integer>(l, u)(rng);
  }}
}

/**
 * Simulate a uniform distribution on unit vectors.
 *
 * - D: Number of dimensions.
 */
function simulate_uniform_unit_vector(D:Integer) -> Real[_] {
  u:Real[D];
  for d in 1..D {
    u[d] <- simulate_gaussian(0.0, 1.0);
  }
  return u/dot(u);
}

/**
 * Simulate an exponential distribution.
 *
 * - λ: Rate.
 */
function simulate_exponential(λ:Real) -> Real {
  assert 0.0 < λ;
  cpp{{
  return std::exponential_distribution<bi::type::Real>(λ)(rng);
  }}
}

/**
 * Simulate an Weibull distribution.
 *
 * - k: Shape.
 * - λ: Scale.
 */
function simulate_weibull(k:Real, λ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < λ;
  cpp{{
  return std::weibull_distribution<bi::type::Real>(k, λ)(rng);
  }}
}

/**
 * Simulate a Gaussian distribution.
 *
 * - μ: Mean.
 * - σ2: Variance.
 */
function simulate_gaussian(μ:Real, σ2:Real) -> Real {
  assert 0.0 <= σ2;
  if (σ2 == 0.0) {
    return μ;
  } else {
    cpp{{
    return std::normal_distribution<bi::type::Real>(μ, std::sqrt(σ2))(rng);
    }}
  }
}

/**
 * Simulate a Student's $t$-distribution.
 *
 * - ν: Degrees of freedom.
 */
function simulate_student_t(ν:Real) -> Real {
  assert 0.0 < ν;
  cpp{{
  return std::student_t_distribution<bi::type::Real>(ν)(rng);
  }}
}

/**
 * Simulate a Student's $t$-distribution with location and scale.
 *
 * - ν: Degrees of freedom.
 * - μ: Location.
 * - σ2: Squared scale.
 */
function simulate_student_t(ν:Real, μ:Real, σ2:Real) -> Real {
  assert 0.0 < ν;
  if σ2 == 0.0 {
    return μ;
  } else {
    return μ + sqrt(σ2)*simulate_student_t(ν);
  }
}

/**
 * Simulate a beta distribution.
 *
 * - α: Shape.
 * - β: Shape.
 */
function simulate_beta(α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;
  
  u:Real <- simulate_gamma(α, 1.0);
  v:Real <- simulate_gamma(β, 1.0);
  
  return u/(u + v);
}

/**
 * Simulate $\chi^2$ distribution.
 *
 * - ν: Degrees of freedom.
 */
function simulate_chi_squared(ν:Real) -> Real {
  assert 0.0 < ν;
  cpp{{
  return std::chi_squared_distribution<bi::type::Real>(ν)(rng);
  }}
}

/**
 * Simulate a gamma distribution.
 *
 * - k: Shape.
 * - θ: Scale.
 */
function simulate_gamma(k:Real, θ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < θ;
  cpp{{
  return std::gamma_distribution<bi::type::Real>(k, θ)(rng);
  }}
}

/**
 * Simulate a Wishart distribution.
 *
 * - Ψ: Scale.
 * - k: Degrees of freedeom.
 */
function simulate_wishart(Ψ:Real[_,_], k:Real) -> Real[_,_] {
  assert rows(Ψ) == columns(Ψ);
  assert k > rows(Ψ) - 1;
  auto p <- rows(Ψ);
  A:Real[p,p];
  
  for i in 1..p {
    for j in 1..p {
      if j == i {
        /* on diagonal */
        A[i,j] <- sqrt(simulate_chi_squared(k - i + 1));
      } else if j < i {
        /* in lower triangle */
        A[i,j] <- simulate_gaussian(0.0, 1.0);
      } else {
        /* in upper triangle */
        A[i,j] <- 0.0;
      }
    }
  }
  auto L <- cholesky(Ψ)*A;
  return L*transpose(L);
}

/**
 * Simulate an inverse-gamma distribution.
 *
 * - α: Shape.
 * - β: Scale.
 */
function simulate_inverse_gamma(α:Real, β:Real) -> Real {
  return 1.0/simulate_gamma(α, 1.0/β);
}

/**
 * Simulate an inverse-Wishart distribution.
 *
 * - Ψ: Scale.
 * - k: Degrees of freedeom.
 */
function simulate_inverse_wishart(Ψ:Real[_,_], k:Real) -> Real[_,_] {
  return inv(llt(simulate_wishart(inv(llt(Ψ)), k)));
}

/**
 * Simulate a normal inverse-gamma distribution.
 *
 * - μ: Mean.
 * - a2: Variance scale.
 * - α: Shape of inverse-gamma on variance.
 * - β: Scale of inverse-gamma on variance.
 */
function simulate_normal_inverse_gamma(μ:Real, a2:Real, α:Real,
    β:Real) -> Real {
  return simulate_student_t(2.0*α, μ, a2*β/α);
}

/**
 * Simulate a beta-bernoulli distribution.
 *
 * - α: Shape.
 * - β: Shape.
 */
function simulate_beta_bernoulli(α:Real, β:Real) -> Boolean {
  assert 0.0 < α;
  assert 0.0 < β;
  
  return simulate_bernoulli(simulate_beta(α, β));
}

/**
 * Simulate a beta-binomial distribution.
 *
 * - n: Number of trials.
 * - α: Shape.
 * - β: Shape.
 */
function simulate_beta_binomial(n:Integer, α:Real, β:Real) -> Integer {
  assert 0 <= n;
  assert 0.0 < α;
  assert 0.0 < β;
  
  return simulate_binomial(n, simulate_beta(α, β));
}

/**
 * Simulate a beta-negative-binomial distribution.
 *
 * - k: Number of successes.
 * - α: Shape.
 * - β: Shape.
 */
function simulate_beta_negative_binomial(k:Integer, α:Real, β:Real) -> Integer {
  assert 0.0 < α;
  assert 0.0 < β;
  assert 0 < k;

  return simulate_negative_binomial(k, simulate_beta(α, β));
}


/**
 * Simulate a gamma-Poisson distribution.
 *
 * - k: Shape.
 * - θ: Scale.
 */
function simulate_gamma_poisson(k:Real, θ:Real) -> Integer {
  assert 0.0 < k;
  assert 0.0 < θ;
  assert k == floor(k);
  
  return simulate_negative_binomial(Integer(k), 1.0/(θ + 1.0));
}

/**
 * Simulate a Lomax distribution.
 *
 * - λ: Scale.
 * - α: Shape.
 */
function simulate_lomax(λ:Real, α:Real) -> Real {
  assert 0.0 < λ;
  assert 0.0 < α;

  u:Real <- simulate_uniform(0.0, 1.0);
  return λ*(pow(u, -1.0/α)-1.0);
}

/**
 * Simulate a Dirichlet-categorical distribution.
 *
 * - α: Concentrations.
 */
function simulate_dirichlet_categorical(α:Real[_]) -> Integer {
  return simulate_categorical(simulate_dirichlet(α));
}

/**
 * Simulate a Dirichlet-multinomial distribution.
 *
 * - n: Number of trials.
 * - α: Concentrations.
 */
function simulate_dirichlet_multinomial(n:Integer, α:Real[_]) -> Integer[_] {
  return simulate_multinomial(n, simulate_dirichlet(α));
}

/**
 * Simulate a categorical distribution with Chinese restaurant process
 * prior.
 *
 * - α: Concentration.
 * - θ: Discount.
 * - n: Enumerated items.
 * - N: Total number of items.
 */
function simulate_crp_categorical(α:Real, θ:Real, n:Integer[_],
    N:Integer) -> Integer {
  assert N >= 0;
  assert sum(n) == N;

  k:Integer <- 0;
  K:Integer <- length(n);
  if (N == 0) {
    /* first component */
    k <- 1;
  } else {
    u:Real <- simulate_uniform(0.0, N + θ);
    U:Real <- K*α + θ;
    if (u < U) {
      /* new component */
      k <- K + 1;
    } else {
      /* existing component */
      while (k < K && u > U) {
        k <- k + 1;
        U <- U + n[k] - α;
      }
    }
  }
  return k;
}

/**
 * Simulate a Gaussian distribution with a normal inverse-gamma prior.
 *
 * - μ: Mean.
 * - a2: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 */
function simulate_normal_inverse_gamma_gaussian(μ:Real, a2:Real,
    α:Real, β:Real) -> Real {
  return simulate_student_t(2.0*α, μ, (β/α)*(1.0 + a2));
}

/**
 * Simulate a Gaussian distribution with a normal inverse-gamma prior.
 *
 * - a: Scale.
 * - μ: Mean.
 * - a2: Variance.
 * - c: Offset.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 */
function simulate_linear_normal_inverse_gamma_gaussian(a:Real, μ:Real,
    a2:Real, c:Real, α:Real, β:Real) -> Real {
  return simulate_student_t(2.0*α, a*μ + c, (β/α)*(1.0 + a*a*a2));
}

/**
 * Simulate a multivariate Gaussian distribution.
 *
 * - μ: Mean.
 * - Σ: Covariance.
 */
function simulate_multivariate_gaussian(μ:Real[_], Σ:Real[_,_]) -> Real[_] {
  auto D <- length(μ);
  z:Real[D];
  for d in 1..D {
    z[d] <- simulate_gaussian(0.0, 1.0);
  }
  return μ + cholesky(Σ)*z;
}

/**
 * Simulate a multivariate Gaussian distribution with independent elements.
 *
 * - μ: Mean.
 * - σ2: Variance.
 */
function simulate_multivariate_gaussian(μ:Real[_], σ2:Real[_]) -> Real[_] {
  auto D <- length(μ);
  z:Real[D];
  for d in 1..D {
    z[d] <- μ[d] + simulate_gaussian(0.0, σ2[d]);
  }
  return z;
}

/**
 * Simulate a multivariate Gaussian distribution with independent elements of
 * identical variance.
 *
 * - μ: Mean.
 * - σ2: Variance.
 */
function simulate_multivariate_gaussian(μ:Real[_], σ2:Real) -> Real[_] {
  auto D <- length(μ);
  auto σ <- sqrt(σ2);
  z:Real[D];
  for d in 1..D {
    z[d] <- μ[d] + σ*simulate_gaussian(0.0, 1.0);
  }
  return z;
}

/**
 * Simulate a multivariate normal inverse-gamma distribution.
 *
 * - ν: Precision times mean.
 * - Λ: Precision.
 * - α: Shape of inverse-gamma on scale.
 * - β: Scale of inverse-gamma on scale.
 */
function simulate_multivariate_normal_inverse_gamma(ν:Real[_], Λ:LLT,
    α:Real, β:Real) -> Real[_] {
  return simulate_multivariate_student_t(2.0*α, solve(Λ, ν), (β/α)*inv(Λ));
}

/**
 * Simulate a multivariate Gaussian distribution with a multivariate normal
 * inverse-gamma prior.
 *
 * - ν: Precision times mean.
 * - Λ: Precision.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 * - γ: Scale accumulator of the inverse-gamma.
 */
function simulate_multivariate_normal_inverse_gamma_multivariate_gaussian(
    ν:Real[_], Λ:LLT, α:Real, γ:Real) -> Real[_] {
  auto μ <- solve(Λ, ν);
  auto β <- γ - 0.5*dot(μ, ν);
  return simulate_multivariate_student_t(2.0*α, μ,
      (β/α)*(identity(rows(Λ)) + inv(Λ)));
}

/**
 * Simulate a Gaussian distribution with a linear transformation of a
 * multivariate linear normal inverse-gamma prior.
 *
 * - A: Scale.
 * - ν: Precision times mean.
 * - Λ: Precision.
 * - c: Offset.
 * - α: Shape of the inverse-gamma.
 * - γ: Scale accumulator of the inverse-gamma.
 */
function simulate_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
    A:Real[_,_], ν:Real[_], Λ:LLT, c:Real[_], α:Real, γ:Real) -> Real[_] {
  auto μ <- solve(Λ, ν);
  auto β <- γ - 0.5*dot(μ, ν);
  return simulate_multivariate_student_t(2.0*α, A*μ + c,
      (β/α)*(identity(rows(A)) + A*solve(Λ, transpose(A))));
}

/**
 * Simulate a matrix Gaussian distribution.
 *
 * - M: Mean.
 * - U: Among-row covariance.
 * - V: Among-column covariance.
 */
function simulate_matrix_gaussian(M:Real[_,_], U:Real[_,_], V:Real[_,_]) ->
    Real[_,_] {
  assert rows(M) == rows(U);
  assert rows(M) == columns(U);
  assert columns(M) == rows(V);
  assert columns(M) == columns(V);
  
  auto N <- rows(M);
  auto P <- columns(M);
  Z:Real[N,P];
  for n in 1..N {
    for p in 1..P {
      Z[n,p] <- simulate_gaussian(0.0, 1.0);
    }
  }
  return M + cholesky(U)*Z*transpose(cholesky(V));
}

/**
 * Simulate a matrix Gaussian distribution with independent columns.
 *
 * - M: Mean.
 * - U: Among-row covariance.
 * - σ2: Among-column variances.
 */
function simulate_matrix_gaussian(M:Real[_,_], U:Real[_,_], σ2:Real[_]) ->
    Real[_,_] {
  assert rows(M) == rows(U);
  assert rows(M) == columns(U);
  assert columns(M) == length(σ2);
  
  auto N <- rows(M);
  auto P <- columns(M);
  Z:Real[N,P];
  for n in 1..N {
    for p in 1..P {
      Z[n,p] <- simulate_gaussian(0.0, 1.0);
    }
  }
  return M + cholesky(U)*Z*diagonal(sqrt(σ2));
}

/**
 * Simulate a matrix Gaussian distribution with independent rows.
 *
 * - M: Mean.
 * - V: Among-column variances.
 */
function simulate_matrix_gaussian(M:Real[_,_], V:Real[_,_]) -> Real[_,_] {
  assert columns(M) == rows(V);
  assert columns(M) == columns(V);
  
  auto N <- rows(M);
  auto P <- columns(M);
  Z:Real[N,P];
  for n in 1..N {
    for p in 1..P {
      Z[n,p] <- simulate_gaussian(0.0, 1.0);
    }
  }
  return M + Z*transpose(cholesky(V));
}

/**
 * Simulate a matrix Gaussian distribution with independent elements.
 *
 * - M: Mean.
 * - σ2: Variances.
 */
function simulate_matrix_gaussian(M:Real[_,_], σ2:Real[_]) -> Real[_,_] {
  assert columns(M) == length(σ2);
  
  auto N <- rows(M);
  auto P <- columns(M);
  X:Real[N,P];
  for n in 1..N {
    for p in 1..P {
      X[n,p] <- simulate_gaussian(M[n,p], σ2[p]);
    }
  }
  return X;
}

/**
 * Simulate a matrix Gaussian distribution with independent elements of
 * identical variance.
 *
 * - M: Mean.
 * - σ2: Variance.
 */
function simulate_matrix_gaussian(M:Real[_,_], σ2:Real) -> Real[_,_] {
  auto N <- rows(M);
  auto P <- columns(M);
  X:Real[N,P];
  for n in 1..N {
    for p in 1..P {
      X[n,p] <- simulate_gaussian(M[n,p], σ2);
    }
  }
  return X;
}

/**
 * Simulate a matrix normal-inverse-gamma distribution.
 *
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - α: Variance shape.
 * - β: Variance scales.
 */
function simulate_matrix_normal_inverse_gamma(N:Real[_,_], Λ:LLT, α:Real,
    β:Real[_]) -> Real[_,_] {
  auto M <- solve(Λ, N);
  auto Σ <- inv(Λ);
  return simulate_matrix_student_t(2.0*α, M, Σ, β/α);
}

/**
 * Simulate a Gaussian distribution with a matrix-normal-inverse-gamma prior.
 *
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - α: Variance shape.
 * - γ: Variance scale accumulators.
 */
function simulate_matrix_normal_inverse_gamma_matrix_gaussian(
    N:Real[_,_], Λ:LLT, α:Real, γ:Real[_]) -> Real[_,_] {
  auto M <- solve(Λ, N);
  auto β <- γ - 0.5*diagonal(transpose(M)*N);
  auto Σ <- identity(rows(M)) + inv(Λ);
  return simulate_matrix_student_t(2.0*α, M, Σ, β/α);
}

/**
 * Simulate a Gaussian distribution with linear transformation of
 * a matrix-normal-inverse-gamma prior.
 *
 * - A: Scale.
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - C: Offset.
 * - α: Variance shape.
 * - γ: Variance scale accumulators.
 */
function simulate_linear_matrix_normal_inverse_gamma_matrix_gaussian(
    A:Real[_,_], N:Real[_,_], Λ:LLT, C:Real[_,_], α:Real, γ:Real[_]) ->
    Real[_,_] {
  auto M <- solve(Λ, N);
  auto β <- γ - 0.5*diagonal(transpose(M)*N);
  auto Σ <- identity(rows(A)) + A*solve(Λ, transpose(A));
  return simulate_matrix_student_t(2.0*α, A*M + C, Σ, β/α);
}

/**
 * Simulate a matrix normal-inverse-Wishart distribution.
 *
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - Ψ: Variance shape.
 * - k: Degrees of freedom.
 */
function simulate_matrix_normal_inverse_wishart(N:Real[_,_], Λ:LLT,
    Ψ:Real[_,_], k:Real) -> Real[_,_] {
  auto p <- columns(N);
  auto M <- solve(Λ, N);
  auto Σ <- inv(Λ)/(k - p + 1.0);
  return simulate_matrix_student_t(k - p + 1.0, M, Σ, Ψ);
}

/**
 * Simulate a Gaussian distribution with matrix-normal-inverse-Wishart prior.
 *
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - Ψ: Variance shape.
 * - k: Degrees of freedom.
 */
function simulate_matrix_normal_inverse_wishart_matrix_gaussian(N:Real[_,_],
    Λ:LLT, Ψ:Real[_,_], k:Real) -> Real[_,_] {
  auto p <- columns(N);
  auto M <- solve(Λ, N);
  auto Σ <- (identity(rows(N)) + inv(Λ))/(k - p + 1.0);
  return simulate_matrix_student_t(k - p + 1.0, M, Σ, Ψ);
}

/**
 * Simulate a Gaussian distribution with linear transformation of a
 * matrix-normal-inverse-Wishart prior.
 *
 * - A: Scale.
 * - N: Precision times mean matrix.
 * - Λ: Precision.
 * - C: Offset.
 * - Ψ: Variance shape.
 * - k: Degrees of freedom.
 */
function simulate_linear_matrix_normal_inverse_wishart_matrix_gaussian(
    A:Real[_,_], N:Real[_,_], Λ:LLT, C:Real[_,_], Ψ:Real[_,_], k:Real) -> Real[_,_] {
  auto p <- columns(N);
  auto M <- solve(Λ, N);
  auto Σ <- (identity(rows(A)) + A*solve(Λ, transpose(A)))/(k - p + 1.0);
  return simulate_matrix_student_t(k - p + 1.0, A*M + C, Σ, Ψ);
}

/**
 * Simulate a multivariate Student's $t$-distribution variate with
 * location and scale.
 *
 * - k: Degrees of freedom.
 * - μ: Mean.
 * - Σ: Covariance.
 */
function simulate_multivariate_student_t(k:Real, μ:Real[_], Σ:Real[_,_]) ->
    Real[_] {
  auto D <- length(μ);
  z:Real[D];
  for d in 1..D {
    z[d] <- simulate_student_t(k);
  }
  return μ + cholesky(Σ)*z;
}

/**
 * Simulate a multivariate Student's $t$-distribution variate with
 * location and diagonal scaling.
 *
 * - k: Degrees of freedom.
 * - μ: Mean.
 * - σ2: Variance.
 */
function simulate_multivariate_student_t(k:Real, μ:Real[_], σ2:Real) ->
    Real[_] {
  auto D <- length(μ);
  auto σ <- sqrt(σ2);
  z:Real[D];
  for d in 1..D {
    z[d] <- μ[d] + σ*simulate_student_t(k);
  }
  return z;
}

/**
 * Simulate a matrix Student's $t$-distribution variate.
 *
 * - k: Degrees of freedom.
 * - M: Mean.
 * - U: Among-row covariance.
 * - V: Among-column covariance.
 */
function simulate_matrix_student_t(k:Real, M:Real[_,_], U:Real[_,_],
    V:Real[_,_]) -> Real[_,_] {
  assert rows(M) == rows(U);
  assert rows(M) == columns(U);
  assert columns(M) == rows(V);
  assert columns(M) == columns(V);
  
  auto N <- rows(M);
  auto P <- columns(M);
  Z:Real[N,P];
  for n in 1..N {
    for p in 1..P {
      Z[n,p] <- simulate_student_t(k);
    }
  }
  return M + cholesky(U)*Z*transpose(cholesky(V));
}

/**
 * Simulate a matrix Student's $t$-distribution variate.
 *
 * - k: Degrees of freedom.
 * - M: Mean.
 * - U: Among-row covariance.
 * - v: Independent within-column covariance.
 */
function simulate_matrix_student_t(k:Real, M:Real[_,_], U:Real[_,_],
    v:Real[_]) -> Real[_,_] {
  assert rows(M) == rows(U);
  assert rows(M) == columns(U);
  assert columns(M) == length(v);
  
  auto N <- rows(M);
  auto P <- columns(M);
  Z:Real[N,P];
  for n in 1..N {
    for p in 1..P {
      Z[n,p] <- simulate_student_t(k);
    }
  }
  return M + cholesky(U)*Z*diagonal(sqrt(v));
}

/**
 * Simulate a multivariate uniform distribution.
 *
 * - l: Lower bound of hyperrectangle.
 * - u: Upper bound of hyperrectangle.
 */
function simulate_independent_uniform(l:Real[_], u:Real[_]) -> Real[_] {
  assert length(l) == length(u);
  D:Integer <- length(l);
  z:Real[D];
  for d in 1..D {
    z[d] <- simulate_uniform(l[d], u[d]);
  }
  return z;
}

/**
 * Simulate a multivariate uniform distribution over integers.
 *
 * - l: Lower bound of hyperrectangle.
 * - u: Upper bound of hyperrectangle.
 */
function simulate_independent_uniform_int(l:Integer[_], u:Integer[_]) -> Integer[_] {
  assert length(l) == length(u);
  D:Integer <- length(l);
  z:Integer[D];
  for d in 1..D {
    z[d] <- simulate_uniform_int(l[d], u[d]);
  }
  return z;
}
