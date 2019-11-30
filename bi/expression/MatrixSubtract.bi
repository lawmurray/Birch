/*
 * Lazy matrix subtraction.
 */
final class MatrixSubtract<Left,Right,Value>(left:Expression<Left>,
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
    return left.value() - right.value();
  }

  function pilot() -> Value {
    return left.pilot() - right.pilot();
  }

  function grad(d:Value) {
    left.grad(d);
    right.grad(-d);
  }

  function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<DelayMatrixGaussian>? {
    y:TransformLinearMatrix<DelayMatrixGaussian>?;
    z:DelayMatrixGaussian?;

    if (y <- left.graftLinearMatrixGaussian())? {
      y!.subtract(right.value());
    } else if (y <- right.graftLinearMatrixGaussian())? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<DelayMatrixGaussian>(
          identity(z!.rows()), z!, -right.value());
    } else if (z <- right.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<DelayMatrixGaussian>(
          diagonal(-1.0, z!.rows()), z!, left.value());
    }
    return y;
  }
  
  function graftLinearMatrixNormalInverseGamma() ->
      TransformLinearMatrix<DelayMatrixNormalInverseGamma>? {
    y:TransformLinearMatrix<DelayMatrixNormalInverseGamma>?;
    z:DelayMatrixNormalInverseGamma?;

    if (y <- left.graftLinearMatrixNormalInverseGamma())? {
      y!.subtract(right.value());
    } else if (y <- right.graftLinearMatrixNormalInverseGamma())? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftMatrixNormalInverseGamma())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseGamma>(
          identity(z!.rows()), z!, -right.value());
    } else if (z <- right.graftMatrixNormalInverseGamma())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseGamma>(
          diagonal(-1.0, z!.rows()), z!, left.value());
    }
    return y;
  }

  function graftLinearMatrixNormalInverseWishart() ->
      TransformLinearMatrix<DelayMatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<DelayMatrixNormalInverseWishart>?;
    z:DelayMatrixNormalInverseWishart?;

    if (y <- left.graftLinearMatrixNormalInverseWishart())? {
      y!.subtract(right.value());
    } else if (y <- right.graftLinearMatrixNormalInverseWishart())? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftMatrixNormalInverseWishart())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseWishart>(
          identity(z!.rows()), z!, -right.value());
    } else if (z <- right.graftMatrixNormalInverseWishart())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseWishart>(
          diagonal(-1.0, z!.rows()), z!, left.value());
    }
    return y;
  }
}

operator (left:Expression<Real[_,_]> - right:Expression<Real[_,_]>) ->
    MatrixSubtract<Real[_,_],Real[_,_],Real[_,_]> {
  m:MatrixSubtract<Real[_,_],Real[_,_],Real[_,_]>(left, right);
  return m;
}

operator (left:Real[_,_] - right:Expression<Real[_,_]>) ->
    MatrixSubtract<Real[_,_],Real[_,_],Real[_,_]> {
  return Boxed(left) - right;
}

operator (left:Expression<Real[_,_]> - right:Real[_,_]) ->
    MatrixSubtract<Real[_,_],Real[_,_],Real[_,_]> {
  return left - Boxed(right);
}
