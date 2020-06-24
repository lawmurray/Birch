/**
 * Lazy matrix pack.
 */
final class MatrixPack<Left,Right,Value>(left:Left, right:Right) <
    MatrixBinaryExpression<Left,Right,Value>(left, right) {  
  override function doRows() -> Integer {
    assert left!.rows() == right!.rows();
    return left!.rows();
  }
  
  override function doColumns() -> Integer {
    return left!.columns() + right!.columns();
  }

  override function doValue() {
    x <- pack(left!.value(), right!.value());
  }

  override function doPilot() {
    x <- pack(left!.pilot(), right!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- pack(left!.move(κ), right!.move(κ));
  }

  override function doGrad() {
    auto R1 <- left!.rows();
    auto R2 <- right!.rows();
    auto C1 <- left!.columns();
    auto C2 <- right!.columns();
    assert R1 == global.rows(d!);
    assert R2 == global.rows(d!);
    
    left!.grad(d![1..R1,1..C1]);
    right!.grad(d![1..R2,(C1 + 1)..(C1 + C2)]);
  }
}

/**
 * Lazy matrix pack.
 */
function pack(left:Expression<Real[_,_]>, right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  assert left!.rows() == right!.rows();
  if left!.isConstant() && right!.isConstant() {
    return box(pack(left!.value(), right!.value()));
  } else {
    m:MatrixPack<Expression<Real[_,_]>,Expression<Real[_,_]>,Real[_,_]>(left, right);
    return m;
  }
}

/**
 * Lazy matrix pack.
 */
function pack(left:Real[_,_], right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  if right!.isConstant() {
    return box(pack(left, right!.value()));
  } else {
    return pack(box(left), right);
  }
}

/**
 * Lazy matrix pack.
 */
function pack(left:Expression<Real[_,_]>, right:Real[_,_]) ->
    Expression<Real[_,_]> {
  if left!.isConstant() {
    return box(pack(left!.value(), right));
  } else {
    return pack(left, box(right));
  }
}
