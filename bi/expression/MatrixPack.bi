/**
 * Lazy matrix pack.
 */
final class MatrixPack(y:Expression<Real[_,_]>, z:Expression<Real[_,_]>) <
    MatrixBinaryExpression<Expression<Real[_,_]>,Expression<Real[_,_]>,
    Real[_,_],Real[_,_],Real[_,_],Real[_,_],Real[_,_]>(y, z) {  
  override function doRows() -> Integer {
    return y!.rows();
  }
  
  override function doColumns() -> Integer {
    return y!.columns() + z!.columns();
  }

  override function doEvaluate(y:Real[_,_], z:Real[_,_]) -> Real[_,_] {
    return pack(y, z);
  }

  override function doEvaluateGradLeft(d:Real[_,_], x:Real[_,_], y:Real[_,_],
      z:Real[_,_]) -> Real[_,_] {
    return d[1..global.rows(y), 1..global.columns(y)];
  }

  override function doEvaluateGradRight(d:Real[_,_], x:Real[_,_], y:Real[_,_],
      z:Real[_,_]) -> Real[_,_] {
    return d[1..global.rows(y), (global.columns(y) + 1)..global.columns(x)];
  }
}

/**
 * Lazy matrix pack.
 */
function pack(y:Expression<Real[_,_]>, z:Expression<Real[_,_]>) ->
    MatrixPack {
  return construct<MatrixPack>(y, z);
}

/**
 * Lazy matrix pack.
 */
function pack(y:Real[_,_], z:Expression<Real[_,_]>) -> MatrixPack {
  return pack(box(y), z);
}

/**
 * Lazy matrix pack.
 */
function pack(y:Expression<Real[_,_]>, z:Real[_,_]) -> MatrixPack {
  return pack(y, box(z));
}
