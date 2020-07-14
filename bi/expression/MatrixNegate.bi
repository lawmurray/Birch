/**
 * Lazy negation.
 */
final class MatrixNegate(y:Expression<Real[_,_]>) <
    MatrixUnaryExpression<Expression<Real[_,_]>,Real[_,_],Real[_,_],
    Real[_,_]>(y) {
  override function doRows() -> Integer {
    return y!.rows();
  }
  
  override function doColumns() -> Integer {
    return y!.columns();
  }

  override function doEvaluate(y:Real[_,_]) -> Real[_,_] {
    return -y;
  }

  override function doEvaluateGrad(d:Real[_,_], x:Real[_,_], y:Real[_,_]) ->
      Real[_,_] {
    return -d;
  }

  override function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<MatrixGaussian>? {
    r:TransformLinearMatrix<MatrixGaussian>?;
    if !hasValue() {
      x1:MatrixGaussian?;

      if (r <- y!.graftLinearMatrixGaussian())? {
        r!.negate();
      } else if (x1 <- y!.graftMatrixGaussian())? {
        auto R <- x1!.rows();
        auto C <- x1!.columns();
        r <- TransformLinearMatrix<MatrixGaussian>(diagonal(box(-1.0), R),
            x1!, box(matrix(0.0, R, C)));
      }
    }
    return r;
  }
  
  override function graftLinearMatrixNormalInverseGamma(compare:Distribution<Real[_]>) ->
      TransformLinearMatrix<MatrixNormalInverseGamma>? {
    r:TransformLinearMatrix<MatrixNormalInverseGamma>?;
    x1:MatrixNormalInverseGamma?;

    if (r <- y!.graftLinearMatrixNormalInverseGamma(compare))? {
      r!.negate();
    } else if (x1 <- y!.graftMatrixNormalInverseGamma(compare))? {
      auto R <- x1!.rows();
      auto C <- x1!.columns();
      r <- TransformLinearMatrix<MatrixNormalInverseGamma>(
          diagonal(box(-1.0), R), x1!, box(matrix(0.0, R, C)));
    }
    return r;
  }

  override function graftLinearMatrixNormalInverseWishart(compare:Distribution<LLT>) ->
      TransformLinearMatrix<MatrixNormalInverseWishart>? {
    r:TransformLinearMatrix<MatrixNormalInverseWishart>?;
    x1:MatrixNormalInverseWishart?;

    if (r <- y!.graftLinearMatrixNormalInverseWishart(compare))? {
      r!.negate();
    } else if (x1 <- y!.graftMatrixNormalInverseWishart(compare))? {
      auto R <- x1!.rows();
      auto C <- x1!.columns();
      r <- TransformLinearMatrix<MatrixNormalInverseWishart>(
          diagonal(box(-1.0), R), x1!, box(matrix(0.0, R, C)));
    }
    return r;
  }
}

/**
 * Lazy negation.
 */
operator (-y:Expression<Real[_,_]>) -> MatrixNegate {
  return construct<MatrixNegate>(y);
}
