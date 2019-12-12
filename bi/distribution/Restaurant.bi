/**
 * Chinese restaurant process.
 */
final class Restaurant(α:Expression<Real>, θ:Expression<Real>) < Distribution<Real[_]> {
  /**
   * Concentration.
   */
  α:Expression<Real> <- α;
  
  /**
   * Strength.
   */
  θ:Expression<Real> <- θ;

  function graft(child:Delay?) {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayRestaurant(future, futureUpdate, α, θ);
    }
  }

  function graftRestaurant(child:Delay?) -> DelayRestaurant? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayRestaurant(future, futureUpdate, α, θ);
    }
    return DelayRestaurant?(delay);
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
