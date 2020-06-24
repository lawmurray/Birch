/**
 * Lazy multivariate add.
 */
final class MultivariateAdd<Left,Right,Value>(left:Left, right:Right) <
    MultivariateBinaryExpression<Left,Right,Value>(left, right) {  
  override function doRows() -> Integer {
    return left!.rows();
  }

  override function doValue() {
    x <- left!.value() + right!.value();
  }

  override function doPilot() {
    x <- left!.pilot() + right!.pilot();
  }

  override function doMove(κ:Kernel) {
    x <- left!.move(κ) + right!.move(κ);
  }

  override function doGrad() {
    left!.grad(d!);
    right!.grad(d!);
  }

  override function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<MultivariateGaussian>? {
    y:TransformLinearMultivariate<MultivariateGaussian>?;
    z:MultivariateGaussian?;

    if (y <- left!.graftLinearMultivariateGaussian())? {
      y!.add(right!);
    } else if (y <- right!.graftLinearMultivariateGaussian())? {
      y!.add(left!);
    } else if (z <- left!.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<MultivariateGaussian>(
          box(identity(z!.rows())), z!, right!);
    } else if (z <- right!.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<MultivariateGaussian>(
          box(identity(z!.rows())), z!, left!);
    }
    return y;
  }
  
  override function graftLinearMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinearMultivariate<MultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<MultivariateNormalInverseGamma>?;
    z:MultivariateNormalInverseGamma?;

    if (y <- left!.graftLinearMultivariateNormalInverseGamma(compare))? {
      y!.add(right!);
    } else if (y <- right!.graftLinearMultivariateNormalInverseGamma(compare))? {
      y!.add(left!);
    } else if (z <- left!.graftMultivariateNormalInverseGamma(compare))? {
      y <- TransformLinearMultivariate<MultivariateNormalInverseGamma>(
          box(identity(z!.rows())), z!, right!);
    } else if (z <- right!.graftMultivariateNormalInverseGamma(compare))? {
      y <- TransformLinearMultivariate<MultivariateNormalInverseGamma>(
          box(identity(z!.rows())), z!, left!);
    }
    return y;
  }
}

/**
 * Lazy multivariate add.
 */
operator (left:Expression<Real[_]> + right:Expression<Real[_]>) ->
    Expression<Real[_]> {
  assert left.rows() == right.rows();
  if left.isConstant() && right.isConstant() {
    return box(vector(left.value() + right.value()));
  } else {
    m:MultivariateAdd<Expression<Real[_]>,Expression<Real[_]>,Real[_]>(left, right);
    return m;
  }
}

/**
 * Lazy multivariate add.
 */
operator (left:Real[_] + right:Expression<Real[_]>) -> Expression<Real[_]> {
  if right.isConstant() {
    return box(vector(left + right.value()));
  } else {
    return box(left) + right;
  }
}

/**
 * Lazy multivariate add.
 */
operator (left:Expression<Real[_]> + right:Real[_]) -> Expression<Real[_]> {
  if left.isConstant() {
    return box(vector(left.value() + right));
  } else {
    return left + box(right);
  }
}
