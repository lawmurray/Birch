/*
 * Affine transformation of a random variable.
 */
class AffineExpression < Expression<Real> {  
  /**
   * Scale.
   */
  a:Real;
  
  /**
   * Random variable.
   */
  x:Random<Real>;
  
  /**
   * Offset.
   */
  c:Real;
  
  /**
   * Value conversion.
   */
  function value() -> Real {
    return a*x.value() + c;
  }
  
  /**
   * Initialize.
   */
  function initialize(a:Real, x:Random<Real>, c:Real) {
    this.a <- a;
    this.x <- x;
    this.c <- c;
  }

  function isGaussian() -> Boolean {
    return x.isGaussian();
  }
  
  function marginalizeGaussian(σ2:Real) -> (Real, Real) {
    return (a*μ_m + c, a*a*σ2_m + σ2);
  }
  
  function conditionGaussian(y:Real, μ_m:Real, σ2_m:Real) {
    x.updateGaussian(update_affine_gaussian_gaussian(y, a, x.μ, x.σ2, μ_m, σ2_m));
  }
}

operator (+x:Random<Real>) -> Random<Real> {
  return x;
}

operator (+x:AffineExpression) -> AffineExpression {
  return x;
}

operator (-x:Random<Real>) -> AffineExpression {
  y:AffineExpression;
  y.initialize(-1.0, x, 0.0);
  return y;
}

operator (-x:AffineExpression) -> AffineExpression {
  y:AffineExpression;
  y.initialize(-x.a, x.x, -x.c);
  return y;
}

operator (x:Random<Real> + c:Real) -> AffineExpression {
  y:AffineExpression;
  y.initialize(1.0, x, c);
  return y;
}

operator (x:AffineExpression + c:Real) -> AffineExpression {
  y:AffineExpression;
  y.initialize(x.a, x.x, x.c + c);
  return y;
}

operator (c:Real + x:Random<Real>) -> AffineExpression {
  return x + c;
}

operator (c:Real + x:AffineExpression) -> AffineExpression {
  return x + c;
}

operator (x:Random<Real> + c:Random<Real>) -> AffineExpression {
  /* favour keeping the random variable on the left */
  return x + c.value();
}

operator (x:Random<Real> - c:Real) -> AffineExpression {
  y:AffineExpression;
  y.initialize(1.0, x, -c);
  return y;
}

operator (x:AffineExpression - c:Real) -> AffineExpression {
  y:AffineExpression;
  y.initialize(x.a, x.x, x.c - c);
  return y;
}

operator (c:Real - x:Random<Real>) -> AffineExpression {
  y:AffineExpression;
  y.initialize(-1.0, x, c);
  return y;
}

operator (c:Real - x:AffineExpression) -> AffineExpression {
  y:AffineExpression;
  y.initialize(-x.a, x.x, c - x.c);
  return y;
}

operator (x:Random<Real> - c:Random<Real>) -> AffineExpression {
  /* favour keeping the random variable on the left */
  return x - c.value();
}

operator (a:Real*x:Random<Real>) -> AffineExpression {
  y:AffineExpression;
  y.initialize(a, x, 0.0);
  return y;
}

operator (a:Real*x:AffineExpression) -> AffineExpression {
  y:AffineExpression;
  y.initialize(a*x.a, x.x, a*x.c);
  return y;
}

operator (x:Random<Real>*a:Real) -> AffineExpression {
  return a*x;
}

operator (x:AffineExpression*a:Real) -> AffineExpression {
  return a*x;
}

operator (a:Random<Real>*x:Random<Real>) -> AffineExpression {
  /* favour keeping the random variable on the right */
  return a.value()*x;
}

operator (x:Random<Real>/a:Real) -> AffineExpression {
  return (1.0/a)*x;
}

operator (x:AffineExpression/a:Real) -> AffineExpression {
  return (1.0/a)*x;
}
