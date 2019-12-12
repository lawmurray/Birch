/**
 * Lazy matrix multiply.
 */
final class MatrixMultiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function rows() -> Integer {
    return left.rows();
  }
  
  function columns() -> Integer {
    return right.columns();
  }
    
  function graft(child:Delay?) -> Expression<Value> {
    return left.graft(child)*right.graft(child);
  }

  function doValue(l:Left, r:Right) -> Value {
    return l*r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*transpose(r), transpose(l)*d);
  }

  function graftLinearMatrixGaussian(child:Delay?) ->
      TransformLinearMatrix<DelayMatrixGaussian>? {
    y:TransformLinearMatrix<DelayMatrixGaussian>?;
    z:DelayMatrixGaussian?;
    
    if (y <- right.graftLinearMatrixGaussian(child))? {
      y!.leftMultiply(left);
    } else if (z <- right.graftMatrixGaussian(child))? {
      y <- TransformLinearMatrix<DelayMatrixGaussian>(left, z!);
    }
    return y;
  }
  
  function graftLinearMatrixNormalInverseGamma(child:Delay?) ->
      TransformLinearMatrix<DelayMatrixNormalInverseGamma>? {
    y:TransformLinearMatrix<DelayMatrixNormalInverseGamma>?;
    z:DelayMatrixNormalInverseGamma?;

    if (y <- right.graftLinearMatrixNormalInverseGamma(child))? {
      y!.leftMultiply(left);
    } else if (z <- right.graftMatrixNormalInverseGamma(child))? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseGamma>(left, z!);
    }
    return y;
  }

  function graftLinearMatrixNormalInverseWishart(child:Delay?) ->
      TransformLinearMatrix<DelayMatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<DelayMatrixNormalInverseWishart>?;
    z:DelayMatrixNormalInverseWishart?;

    if (y <- right.graftLinearMatrixNormalInverseWishart(child))? {
      y!.leftMultiply(left);
    } else if (z <- right.graftMatrixNormalInverseWishart(child))? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseWishart>(left, z!);
    }
    return y;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Expression<Real[_,_]>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  assert left.columns() == right.rows();
  m:MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]>(left, right);
  return m;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real[_,_]*right:Expression<Real[_,_]>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return Boxed(left)*right;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Real[_,_]) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return left*Boxed(right);
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real>*right:Expression<Real[_,_]>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  m:MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]>(diagonal(left,
      right.rows()), right);
  return m;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real*right:Expression<Real[_,_]>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return diagonal(Boxed(left), right.rows())*right;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real>*right:Real[_,_]) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return diagonal(left, rows(right))*Boxed(right);
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Expression<Real>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  m:MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]>(diagonal(right,
      left.rows()), left);
  return m;
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Real[_,_]*right:Expression<Real>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return diagonal(right, right.rows())*Boxed(left);
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_,_]>*right:Real) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return diagonal(Boxed(right), left.rows())*left;
}
