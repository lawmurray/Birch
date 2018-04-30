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

  function doGraft() -> DelayValue<Real[_]>? {
    return DelayRestaurant(this, α, θ);
  }

  function doGraftRestaurant() -> DelayRestaurant? {
    return DelayRestaurant(this, α, θ);
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
