/**
 * Lazy multivariate multiply.
 */
final class MultivariateMultiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  override function rows() -> Integer {
    return left.rows();
  }

  override function computeValue(l:Left, r:Right) -> Value {
    return l*r;
  }

  override function computeGrad(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*transpose(r), transpose(l)*d);
  }

  override function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<MultivariateGaussian>? {
    y:TransformLinearMultivariate<MultivariateGaussian>?;
    z:MultivariateGaussian?;
    
    if (y <- right.graftLinearMultivariateGaussian())? {
      y!.leftMultiply(left);
    } else if (z <- right.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<MultivariateGaussian>(left, z!);
    }
    return y;
  }
  
  override function graftLinearMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinearMultivariate<MultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<MultivariateNormalInverseGamma>?;
    z:MultivariateNormalInverseGamma?;

    if (y <- right.graftLinearMultivariateNormalInverseGamma(compare))? {
      y!.leftMultiply(left);
    } else if (z <- right.graftMultivariateNormalInverseGamma(compare))? {
      y <- TransformLinearMultivariate<MultivariateNormalInverseGamma>(left, z!);
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
    m:MultivariateMultiply<Real[_,_],Real[_],Real[_]>(left, right);
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
    return Boxed(left)*right;
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_,_]>*right:Real[_]) -> Expression<Real[_]> {
  if left.isConstant() {
    return box(vector(left.value()*right));
  } else {
    return left*Boxed(right);
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real>*right:Expression<Real[_]>) ->
    Expression<Real[_]> {
  if left.isConstant() && right.isConstant() {
    return box(vector(left.value()*right.value()));
  } else {
    return diagonal(left, right.rows())*right;
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Real*right:Expression<Real[_]>) -> Expression<Real[_]> {
  if right.isConstant() {
    return box(vector(left*right.value()));
  } else {
    return Boxed(left)*right;
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real>*right:Real[_]) -> Expression<Real[_]> {
  if left.isConstant() {
    return box(vector(left.value()*right));
  } else {
    return left*Boxed(right);
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_]>*right:Expression<Real>) ->
    Expression<Real[_]> {
  if left.isConstant() && right.isConstant() {
    return box(vector(left.value()*right.value()));
  } else {
    return diagonal(right, left.rows())*left;
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Real[_]*right:Expression<Real>) -> Expression<Real[_]> {
  if right.isConstant() {
    return box(vector(left*right.value()));
  } else {
    return Boxed(left)*right;
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_]>*right:Real) -> Expression<Real[_]> {
  if left.isConstant() {
    return box(vector(left.value()*right));
  } else {
    return left*Boxed(right);
  }
}
