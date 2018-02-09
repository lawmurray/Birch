/**
 * (Chinese) restaurant process.
 */
class RestaurantProcess < Random<Real[_]> {
  /**
   * Concentration parameter.
   */
  α:Real;
  
  /**
   * Strength parameter.
   */
  θ:Real;

  /**
   * Number of samples drawn in each component.
   */
  n:Integer[_];

  /**
   * Number of components enumerated.
   */
  K:Integer;

  /**
   * Number of samples drawn.
   */
  N:Integer;
  
  /**
   * Constructor.
   *
   *   - α: Strength.
   *   - θ: Concentration.
   */
  function initialize(α:Real, θ:Real) {
    /* pre-condition */
    assert (0.0 <= α && α < 1.0 && θ > -α) || (α < 0.0 && round(-θ/α) == -θ/α);

    this.α <- α;
    this.θ <- θ;
    this.K <- 0;
    this.N <- 0;
  }

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

function RestaurantProcess(α:Real, θ:Real) -> RestaurantProcess {
  x:RestaurantProcess;
  x.initialize(α, θ);
  return x;
}
