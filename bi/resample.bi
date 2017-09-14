import primitive;
import math.simulate;

/**
 * Sample an ancestry vector for a vector of log-weights.
 */
function ancestors(w:Real[_]) -> Integer[_] {
  N:Integer <- length(w);
  a:Integer[N];
  O:Integer[N];
  
  O <- systematic_cumulative_offspring(cumulative_weights(w));
  a <- cumulative_offspring_to_ancestors(O);
  
  return permute_ancestors(a);
}

/**
 * Sample a single ancestor for a vector of log-weights.
 */
function ancestor(w:Real[_]) -> Integer {
  N:Integer <- length(w);
  W:Real[N];
  u:Real;
  n:Integer;
  
  W <- cumulative_weights(w);
  u <- simulate_uniform(0.0, W[N]);
  n <- 1;
  while (W[n] < u) {
    n <- n + 1;
  }
  return n;
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
    r <- Real(N)*W[n]/W[N];
    O[n] <- min(N, Integer(floor(r + u)));
  }
  return O;
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
  mx:Real <- max(w);
  W:Real[N];
  
  W[1] <- exp(w[1] - mx);
  for (n:Integer in 2..N) {
    W[n] <- W[n - 1] + exp(w[n] - mx);
  }
  return W;
}
