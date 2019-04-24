/**
 * Exponentiate and sum a vector, return the logarithm of the sum.
 */
function log_sum_exp(x:Real[_]) -> Real {
  assert length(x) > 0;
  mx:Real <- max(x);
  r:Real <- 0.0;
  for (n:Integer in 1..length(x)) {
    r <- r + exp(x[n] - mx);
  }
  return mx + log(r);
}

/**
 * Exponentiate a vector and normalize to sum to one.
 */
function norm_exp(x:Real[_]) -> Real[_] {
  assert length(x) > 0;
  
  auto mx <- max(x);
  auto r <- 0.0;
  for auto n in 1..length(x) {
    r <- r + exp(x[n] - mx);
  }
  auto W <- mx + log(r);
  return transform(x, @(w:Real) -> Real { return exp(w - W); });
}

/**
 * Resample
 *
 * - w: Log weights.
 *
 * Return: a pair the vectors, the first is the vector of ancestor indices,
 * the second the vector of offspring counts.
 */
function resample(w:Real[_]) -> (Integer[_], Integer[_]) {
  auto O <- systematic_cumulative_offspring(cumulative_weights(w));
  auto a <- cumulative_offspring_to_ancestors(O);
  auto o <- cumulative_offspring_to_offspring(O);
  return (a, o);
}

/**
 * Sample an ancestry vector for a log-weight vector using multinomial
 * resampling.
 */
function multinomial_resample(w:Real[_]) -> (Integer[_], Integer[_]) {
  auto o <- simulate_multinomial(length(w), norm_exp(w));
  auto a <- offspring_to_ancestors(o);
  return (a, o);
}

/**
 * Sample a conditional ancestry vector for a log-weight vector using
 * multinomial resampling.
 */
function multinomial_conditional_resample(w:Real[_]) -> (Integer[_], Integer[_]) {
  auto N <- length(w);
  auto o <- simulate_multinomial(N - 1, norm_exp(w));
  o[N] <- o[N] + 1;
  auto a <- offspring_to_ancestors(o);
  return (a, o);
}

/**
 * Sample a single ancestor for a cumulative weight vector.
 */
function cumulative_ancestor(W:Real[_]) -> Integer {
  N:Integer <- length(W);
  u:Real;
  n:Integer;

  assert W[N] > 0.0;
  u <- simulate_uniform(0.0, W[N]);
  n <- 1;
  while (W[n] < u) {
    n <- n + 1;
  }
  return n;
}

/**
 * Sample a single ancestor for a log-weight vector. If the sum of
 * weights is zero, returns zero.
 */
function ancestor(w:Real[_]) -> Integer {
  N:Integer <- length(w);
  W:Real[N];

  W <- cumulative_weights(w);
  if (W[N] > 0.0) {
    return cumulative_ancestor(W);
  } else {
    return 0;
  }
}

/**
 * Systematic resampling.
 */
function systematic_cumulative_offspring(W:Real[_]) -> Integer[_] {
  N:Integer <- length(W);
  u:Real;
  O:Integer[N];
  r:Real;

  u <- simulate_uniform(0.0, 1.0);
  for (n:Integer in 1..N) {
    r <- N*W[n]/W[N];
    O[n] <- min(N, Integer(floor(r + u)));
  }
  return O;
}

/**
 * Convert an offspring vector into an ancestry vector.
 */
function offspring_to_ancestors(o:Integer[_]) -> Integer[_] {
  auto N <- length(o);
  auto i <- 1;
  a:Integer[N];
  for auto n in 1..N {
    for auto j in 1..o[n] {
      a[i] <- n;
      i <- i + 1;
    }
  }
  assert i == N + 1;
  return a;
}

/**
 * Convert a cumulative offspring vector into an ancestry vector.
 */
function cumulative_offspring_to_ancestors(O:Integer[_]) -> Integer[_] {
  N:Integer <- length(O);
  a:Integer[N];
  start:Integer;
  o:Integer;
  for (n:Integer in 1..N) {
    if (n == 1) {
      start <- 0;
    } else {
      start <- O[n - 1];
    }
    o <- O[n] - start;
    for (j:Integer in 1..o) {
      a[start + j] <- n;
    }
  }
  return a;
}

/**
 * Convert a cumulative offspring vector into an offspring vector.
 */
function cumulative_offspring_to_offspring(O:Integer[_]) -> Integer[_] {
  return adjacent_difference(O, @(x:Integer, y:Integer) -> Integer { return x - y; });
}

/**
 * Permute an ancestry vector to ensure that, when a particle survives, at
 * least one of its instances remains in the same place.
 */
function permute_ancestors(a:Integer[_]) -> Integer[_] {
  N:Integer <- length(a);
  b:Integer[N];
  c:Integer;
  
  b <- a;
  for (n:Integer in 1..N) {
    c <- b[n];
    if (c != n && b[c] != c) {
      b[n] <- b[c];
      b[c] <- c;
      n <- n - 1;
    }
  }
  return b;
}

/**
 * Compute the cumulative weight vector from the log-weight vector.
 */
function cumulative_weights(w:Real[_]) -> Real[_] {
  N:Integer <- length(w);
  W:Real[N];
  
  if (N > 0) {
    mx:Real <- max(w);
    W[1] <- exp(w[1] - mx);
    for (n:Integer in 2..N) {
      W[n] <- W[n - 1] + exp(w[n] - mx);
    }
  }
  return W;
}

/**
 * Effective sample size (ESS) of the log-weight vector.
 */
function ess(w:Real[_]) -> Real {
  if (length(w) == 0) {
    return 0.0;
  } else {
    W:Real <- 0.0;
    W2:Real <- 0.0;
    m:Real <- max(w);
    v:Real;
    
    for (n:Integer in 1..length(w)) {
      v <- exp(w[n] - m);
      W <- W + v;
      W2 <- W2 + v*v;
    }
    return W*W/W2;
  }
}
