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

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayRestaurant(x, α, θ);
    }
  }

  function graftRestaurant() -> DelayRestaurant? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayRestaurant(x, α, θ);
    }
    return DelayRestaurant?(delay);
  }

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "Restaurant");
      buffer.set("α", α.value());
      buffer.set("θ", θ.value());
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
