/**
 * Chinese restaurant process.
 */
class Restaurant(α:Expression<Real>, θ:Expression<Real>) < Random<Real[_]> {
  /**
   * Concentration.
   */
  α:Expression<Real> <- α;
  
  /**
   * Strength.
   */
  θ:Expression<Real> <- θ;

  function graft() -> Delay? {
    if (delay?) {
      return delay;
    } else {
      return DelayRestaurant(this, α, θ);
    }
  }

  function graftRestaurant() -> DelayRestaurant? {
    if (delay?) {
      return DelayRestaurant?(graftRestaurant());
    } else {
      return DelayRestaurant(this, α, θ);
    }
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
  return Restaurant(α, Boxed(θ));
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Real, θ:Expression<Real>) -> Restaurant {
  return Restaurant(Boxed(α), θ);
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Real, θ:Real) -> Restaurant {
  return Restaurant(Boxed(α), Boxed(θ));
}
