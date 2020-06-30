/**
 * Lazy negation.
 */
final class MatrixNegate(x:Expression<Real[_,_]>) <
    MatrixUnaryExpression<Expression<Real[_,_]>,Real[_,_]>(x) {
  override function doRows() -> Integer {
    return single!.rows();
  }
  
  override function doColumns() -> Integer {
    return single!.columns();
  }

  override function doValue() {
    x <- -single!.value();
  }

  override function doPilot() {
    x <- -single!.pilot();
  }

  override function doMove(κ:Kernel) {
    x <- -single!.move(κ);
  }

  override function doGrad() {
    single!.grad(-d!);
  }

  override function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<MatrixGaussian>? {
    y:TransformLinearMatrix<MatrixGaussian>?;
    if !hasValue() {
      z:MatrixGaussian?;

      if (y <- single!.graftLinearMatrixGaussian())? {
        y!.negate();
      } else if (z <- single!.graftMatrixGaussian())? {
        auto R <- z!.rows();
        auto C <- z!.columns();
        y <- TransformLinearMatrix<MatrixGaussian>(diagonal(box(-1.0), R), z!,
            box(matrix(0.0, R, C)));
      }
    }
    return y;
  }
  
  override function graftLinearMatrixNormalInverseGamma(compare:Distribution<Real[_]>) ->
      TransformLinearMatrix<MatrixNormalInverseGamma>? {
    y:TransformLinearMatrix<MatrixNormalInverseGamma>?;
    z:MatrixNormalInverseGamma?;

    if (y <- single!.graftLinearMatrixNormalInverseGamma(compare))? {
      y!.negate();
    } else if (z <- single!.graftMatrixNormalInverseGamma(compare))? {
      auto R <- z!.rows();
      auto C <- z!.columns();
      y <- TransformLinearMatrix<MatrixNormalInverseGamma>(
          diagonal(box(-1.0), R), z!, box(matrix(0.0, R, C)));
    }
    return y;
  }

  override function graftLinearMatrixNormalInverseWishart(compare:Distribution<LLT>) ->
      TransformLinearMatrix<MatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<MatrixNormalInverseWishart>?;
    z:MatrixNormalInverseWishart?;

    if (y <- single!.graftLinearMatrixNormalInverseWishart(compare))? {
      y!.negate();
    } else if (z <- single!.graftMatrixNormalInverseWishart(compare))? {
      auto R <- z!.rows();
      auto C <- z!.columns();
      y <- TransformLinearMatrix<MatrixNormalInverseWishart>(
          diagonal(box(-1.0), R), z!, box(matrix(0.0, R, C)));
    }
    return y;
  }
}

/**
 * Lazy negation.
 */
operator (-x:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(-x.value()));
  } else {
    m:MatrixNegate(x);
    return m;
  }
}

/**
 * Lazy negation.
 */
operator (-x:Expression<LLT>) -> Expression<Real[_,_]> {
  return -matrix(x);
}
