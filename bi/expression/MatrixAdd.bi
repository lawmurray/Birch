/*
 * Lazy multivariate addition.
 */
final class MatrixAdd<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < Expression<Value> {  
  /**
   * Left operand.
   */
  left:Expression<Left> <- left;
  
  /**
   * Right operand.
   */
  right:Expression<Right> <- right;

  function value() -> Value {
    return left.value() + right.value();
  }

  function pilot() -> Value {
    return left.pilot() + right.pilot();
  }

  function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<DelayMatrixGaussian>? {
    y:TransformLinearMatrix<DelayMatrixGaussian>?;
    z:DelayMatrixGaussian?;

    if (y <- left.graftLinearMatrixGaussian())? {
      y!.add(right.value());
    } else if (y <- right.graftLinearMatrixGaussian())? {
      y!.add(left.value());
    } else if (z <- left.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<DelayMatrixGaussian>(
          identity(z!.rows()), z!, right.value());
    } else if (z <- right.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<DelayMatrixGaussian>(
          identity(z!.rows()), z!, left.value());
    }
    return y;
  }
  
  function graftLinearMatrixNormalInverseGamma() ->
      TransformLinearMatrix<DelayMatrixNormalInverseGamma>? {
    y:TransformLinearMatrix<DelayMatrixNormalInverseGamma>?;
    z:DelayMatrixNormalInverseGamma?;

    if (y <- left.graftLinearMatrixNormalInverseGamma())? {
      y!.add(right.value());
    } else if (y <- right.graftLinearMatrixNormalInverseGamma())? {
      y!.add(left.value());
    } else if (z <- left.graftMatrixNormalInverseGamma())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseGamma>(
          identity(z!.rows()), z!, right.value());
    } else if (z <- right.graftMatrixNormalInverseGamma())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseGamma>(
          identity(z!.rows()), z!, left.value());
    }
    return y;
  }

  function graftLinearMatrixNormalInverseWishart() ->
      TransformLinearMatrix<DelayMatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<DelayMatrixNormalInverseWishart>?;
    z:DelayMatrixNormalInverseWishart?;

    if (y <- left.graftLinearMatrixNormalInverseWishart())? {
      y!.add(right.value());
    } else if (y <- right.graftLinearMatrixNormalInverseWishart())? {
      y!.add(left.value());
    } else if (z <- left.graftMatrixNormalInverseWishart())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseWishart>(
          identity(z!.rows()), z!, right.value());
    } else if (z <- right.graftMatrixNormalInverseWishart())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseWishart>(
          identity(z!.rows()), z!, left.value());
    }
    return y;
  }
}

operator (left:Expression<Real[_,_]> + right:Expression<Real[_,_]>) ->
    MatrixAdd<Real[_,_],Real[_,_],Real[_,_]> {
  m:MatrixAdd<Real[_,_],Real[_,_],Real[_,_]>(left, right);
  return m;
}

operator (left:Real[_,_] + right:Expression<Real[_,_]>) ->
    MatrixAdd<Real[_,_],Real[_,_],Real[_,_]> {
  return Boxed(left) + right;
}

operator (left:Expression<Real[_,_]> + right:Real[_,_]) ->
    MatrixAdd<Real[_,_],Real[_,_],Real[_,_]> {
  return left + Boxed(right);
}
