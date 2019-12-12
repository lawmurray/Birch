/**
 * Lazy matrix subtrac.
 */
final class MatrixSubtract<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function rows() -> Integer {
    assert left.rows() == right.rows();
    return left.rows();
  }
  
  function columns() -> Integer {
    assert left.rows() == right.rows();
    return left.columns();
  }

  function graft(child:Delay?) -> Expression<Value> {
    return left.graft(child) - right.graft(child);
  }


  function doValue(l:Left, r:Right) -> Value {
    return l - r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d, -d);
  }

  function graftLinearMatrixGaussian(child:Delay?) ->
      TransformLinearMatrix<DelayMatrixGaussian>? {
    y:TransformLinearMatrix<DelayMatrixGaussian>?;
    z:DelayMatrixGaussian?;

    if (y <- left.graftLinearMatrixGaussian(child))? {
      y!.subtract(right);
    } else if (y <- right.graftLinearMatrixGaussian(child))? {
      y!.negateAndAdd(left);
    } else if (z <- left.graftMatrixGaussian(child))? {
      y <- TransformLinearMatrix<DelayMatrixGaussian>(
          Boxed(identity(z!.rows())), z!, -right);
    } else if (z <- right.graftMatrixGaussian(child))? {
      y <- TransformLinearMatrix<DelayMatrixGaussian>(
          Boxed(diagonal(-1.0, z!.rows())), z!, left);
    }
    return y;
  }
  
  function graftLinearMatrixNormalInverseGamma(child:Delay?) ->
      TransformLinearMatrix<DelayMatrixNormalInverseGamma>? {
    y:TransformLinearMatrix<DelayMatrixNormalInverseGamma>?;
    z:DelayMatrixNormalInverseGamma?;

    if (y <- left.graftLinearMatrixNormalInverseGamma(child))? {
      y!.subtract(right);
    } else if (y <- right.graftLinearMatrixNormalInverseGamma(child))? {
      y!.negateAndAdd(left);
    } else if (z <- left.graftMatrixNormalInverseGamma(child))? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseGamma>(
          Boxed(identity(z!.rows())), z!, -right);
    } else if (z <- right.graftMatrixNormalInverseGamma(child))? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseGamma>(
          Boxed(diagonal(-1.0, z!.rows())), z!, left);
    }
    return y;
  }

  function graftLinearMatrixNormalInverseWishart(child:Delay?) ->
      TransformLinearMatrix<DelayMatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<DelayMatrixNormalInverseWishart>?;
    z:DelayMatrixNormalInverseWishart?;

    if (y <- left.graftLinearMatrixNormalInverseWishart(child))? {
      y!.subtract(right);
    } else if (y <- right.graftLinearMatrixNormalInverseWishart(child))? {
      y!.negateAndAdd(left);
    } else if (z <- left.graftMatrixNormalInverseWishart(child))? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseWishart>(
          Boxed(identity(z!.rows())), z!, -right);
    } else if (z <- right.graftMatrixNormalInverseWishart(child))? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseWishart>(
          Boxed(diagonal(-1.0, z!.rows())), z!, left);
    }
    return y;
  }
}

/**
 * Lazy matrix subtract.
 */
operator (left:Expression<Real[_,_]> - right:Expression<Real[_,_]>) ->
    MatrixSubtract<Real[_,_],Real[_,_],Real[_,_]> {
  assert left.rows() == right.rows();
  assert left.columns() == right.columns();
  m:MatrixSubtract<Real[_,_],Real[_,_],Real[_,_]>(left, right);
  return m;
}

/**
 * Lazy matrix subtract.
 */
operator (left:Real[_,_] - right:Expression<Real[_,_]>) ->
    MatrixSubtract<Real[_,_],Real[_,_],Real[_,_]> {
  return Boxed(left) - right;
}

/**
 * Lazy matrix subtract.
 */
operator (left:Expression<Real[_,_]> - right:Real[_,_]) ->
    MatrixSubtract<Real[_,_],Real[_,_],Real[_,_]> {
  return left - Boxed(right);
}
