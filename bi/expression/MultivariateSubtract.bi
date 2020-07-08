/**
 * Lazy multivariate subtract.
 */
final class MultivariateSubtract<Left,Right,Value>(left:Left, right:Right) <
    MultivariateBinaryExpression<Left,Right,Value>(left, right) {  
  override function doRows() -> Integer {
    return left!.rows();
  }

  override function doValue() {
    x <- left!.value() - right!.value();
  }

  override function doPilot() {
    x <- left!.pilot() - right!.pilot();
  }

  override function doMove(κ:Kernel) {
    x <- left!.move(κ) - right!.move(κ);
  }

  override function doGrad() {
    left!.grad(d!);
    right!.grad(-d!);
  }

  override function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<MultivariateGaussian>? {
    y:TransformLinearMultivariate<MultivariateGaussian>?;
    if !hasValue() {
      z:MultivariateGaussian?;

      if (y <- left!.graftLinearMultivariateGaussian())? {
        y!.subtract(right!);
      } else if (y <- right!.graftLinearMultivariateGaussian())? {
        y!.negateAndAdd(left!);
      } else if (z <- left!.graftMultivariateGaussian())? {
        y <- TransformLinearMultivariate<MultivariateGaussian>(box(identity(z!.rows())), z!, -right!);
      } else if (z <- right!.graftMultivariateGaussian())? {
        y <- TransformLinearMultivariate<MultivariateGaussian>(box(diagonal(-1.0, z!.rows())), z!, left!);
      }
    }
    return y;
  }
  
  override function graftLinearMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinearMultivariate<MultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<MultivariateNormalInverseGamma>?;
    if !hasValue() {
      z:MultivariateNormalInverseGamma?;

      if (y <- left!.graftLinearMultivariateNormalInverseGamma(compare))? {
        y!.subtract(right!);
      } else if (y <- right!.graftLinearMultivariateNormalInverseGamma(compare))? {
        y!.negateAndAdd(left!);
      } else if (z <- left!.graftMultivariateNormalInverseGamma(compare))? {
        y <- TransformLinearMultivariate<MultivariateNormalInverseGamma>(box(identity(z!.rows())), z!, -right!);
      } else if (z <- right!.graftMultivariateNormalInverseGamma(compare))? {
        y <- TransformLinearMultivariate<MultivariateNormalInverseGamma>(box(diagonal(-1.0, z!.rows())), z!, left!);
      }
    }
    return y;
  }
}

/**
 * Lazy multivariate subtract.
 */
operator (left:Expression<Real[_]> - right:Expression<Real[_]>) ->
    Expression<Real[_]> {
  assert left.rows() == right.rows();
  if left.isConstant() && right.isConstant() {
    return box(vector(left.value() - right.value()));
  } else {
    return construct<MultivariateSubtract<Expression<Real[_]>,Expression<Real[_]>,Real[_]>>(left, right);
  }
}

/**
 * Lazy multivariate subtract.
 */
operator (left:Real[_] - right:Expression<Real[_]>) -> Expression<Real[_]> {
  if right.isConstant() {
    return box(vector(left - right.value()));
  } else {
    return box(left) - right;
  }
}

/**
 * Lazy multivariate subtract.
 */
operator (left:Expression<Real[_]> - right:Real[_]) -> Expression<Real[_]> {
  if left.isConstant() {
    return box(vector(left.value() - right));
  } else {
    return left - box(right);
  }
}
