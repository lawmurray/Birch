/**
 * Lazy multivariate multiply.
 */
final class MultivariateMultiply<Left,Right,Value>(left:Left, right:Right) <
    MultivariateBinaryExpression<Left,Right,Value>(left, right) {  
  override function doRows() -> Integer {
    return left!.rows();
  }

  override function doValue() {
    x <- left!.value()*right!.value();
  }

  override function doPilot() {
    x <- left!.pilot()*right!.pilot();
  }

  override function doMove(κ:Kernel) {
    x <- left!.move(κ)*right!.move(κ);
  }

  override function doGrad() {
    left!.grad(d!*transpose(right!.get()));
    right!.grad(transpose(left!.get())*d!);
  }

  override function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<MultivariateGaussian>? {
    y:TransformLinearMultivariate<MultivariateGaussian>?;
    if !hasValue() {
      z:MultivariateGaussian?;
    
      if (y <- right!.graftLinearMultivariateGaussian())? {
        y!.leftMultiply(matrix(left!));
      } else if (z <- right!.graftMultivariateGaussian())? {
        y <- TransformLinearMultivariate<MultivariateGaussian>(matrix(left!), z!);
      }
    }
    return y;
  }
  
  override function graftLinearMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinearMultivariate<MultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<MultivariateNormalInverseGamma>?;
    if !hasValue() {
      z:MultivariateNormalInverseGamma?;

      if (y <- right!.graftLinearMultivariateNormalInverseGamma(compare))? {
        y!.leftMultiply(matrix(left!));
      } else if (z <- right!.graftMultivariateNormalInverseGamma(compare))? {
        y <- TransformLinearMultivariate<MultivariateNormalInverseGamma>(matrix(left!), z!);
      }
    }
    return y;
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_,_]>*right:Expression<Real[_]>) ->
    Expression<Real[_]> {
  assert left.columns() == right.rows();
  if left.isConstant() && right.isConstant() {
    return box(vector(left.value()*right.value()));
  } else {
    m:MultivariateMultiply<Expression<Real[_,_]>,Expression<Real[_]>,Real[_]>(left, right);
    return m;
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Real[_,_]*right:Expression<Real[_]>) -> Expression<Real[_]> {
  if right.isConstant() {
    return box(vector(left*right.value()));
  } else {
    return box(left)*right;
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_,_]>*right:Real[_]) -> Expression<Real[_]> {
  if left.isConstant() {
    return box(vector(left.value()*right));
  } else {
    return left*box(right);
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<LLT>*right:Expression<Real[_]>) ->
    Expression<Real[_]> {
  assert left.columns() == right.rows();
  if left.isConstant() && right.isConstant() {
    return box(vector(left.value()*right.value()));
  } else {
    m:MultivariateMultiply<Expression<LLT>,Expression<Real[_]>,Real[_]>(left, right);
    return m;
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:LLT*right:Expression<Real[_]>) -> Expression<Real[_]> {
  if right.isConstant() {
    return box(vector(left*right.value()));
  } else {
    return box(left)*right;
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<LLT>*right:Real[_]) -> Expression<Real[_]> {
  if left.isConstant() {
    return box(vector(left.value()*right));
  } else {
    return left*box(right);
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real>*right:Expression<Real[_]>) ->
    Expression<Real[_]> {
  return diagonal(left, right.rows())*right;
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Real*right:Expression<Real[_]>) -> Expression<Real[_]> {
  return box(left)*right;
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real>*right:Real[_]) -> Expression<Real[_]> {
  return left*box(right);
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_]>*right:Expression<Real>) ->
    Expression<Real[_]> {
  return right*left;
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Real[_]*right:Expression<Real>) -> Expression<Real[_]> {
  return right*left;
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_]>*right:Real) -> Expression<Real[_]> {
  return right*left;
}
