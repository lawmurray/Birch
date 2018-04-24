/**
 * Chinese restaurant process.
 */
class Restaurant(α:Expression<Real>, θ:Expression<Real>) < Random<Real[_]> {
  /**
   * Concentration parameter.
   */
  α:Expression<Real> <- α;
  
  /**
   * Strength parameter.
   */
  θ:Expression<Real> <- θ;

  /**
   * Number of samples drawn in each component.
   */
  n:Integer[_];

  /**
   * Number of components enumerated.
   */
  K:Integer <- 0;

  /**
   * Number of samples drawn.
   */
  N:Integer <- 0;
  
  function update(k:Integer) {
    assert k <= K + 1;
    if (k == K + 1) {
      if (k > length(n)) {
        n1:Integer[max(1, 2*length(n))];
        n1[1..K] <- n;
        n <- n1;
      }
      n[K + 1] <- 1;
      K <- K + 1;
    } else {
      n[k] <- n[k] + 1;
    }
    N <- N + 1;
  }
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Expression<Real>, θ:Expression<Real>) -> Restaurant {
  x:Restaurant(α, θ);
  return x;
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Expression<Real>, θ:Real) -> Restaurant {
  return Restaurant(α, Literal(θ));
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Real, θ:Expression<Real>) -> Restaurant {
  return Restaurant(Literal(α), θ);
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Real, θ:Real) -> Restaurant {
  return Restaurant(Literal(α), Literal(θ));
}
