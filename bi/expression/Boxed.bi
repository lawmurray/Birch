/**
 * Boxed value.
 */
final class Boxed<Value> < Expression<Value> {  
  override function rows() -> Integer {
    return global.rows(get());
  }

  override function columns() -> Integer {
    return global.columns(get());
  }

  override function doValue() {
    //
  }

  override function doMakeConstant() {
    //
  }

  override function doPilot() {
    //
  }

  override function doRestoreCount() {
    //
  }

  override function doMove(κ:Kernel) {
    //
  }
  
  override function doClearGrad() {
    //
  }

  override function doGrad() {
    //
  }

  override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    return nil;
  }
}

/**
 * Create a Boxed value.
 */
function Boxed<Value>(x:Value) -> Expression<Value> {
  o:Boxed<Value>;
  o.x <- x;
  o.makeConstant();
  return o;
}

/**
 * Box a value.
 */
function box<Value>(x:Value) -> Expression<Value> {
  return Boxed(x);
}

/**
 * Box elements of a tuple.
 */
function box<Value1,Value2>(x:(Value1,Value2)) ->
    (Expression<Value1>, Expression<Value2>) {
  x1:Value1?;
  x2:Value2?;
  (x1, x2) <- x;
  return (box(x1!), box(x2!));
}

/**
 * Box elements of a tuple.
 */
function box<Value1,Value2,Value3>(x:(Value1,Value2,Value3)) ->
    (Expression<Value1>, Expression<Value2>, Expression<Value3>) {
  x1:Value1?;
  x2:Value2?;
  x3:Value3?;
  (x1, x2, x3) <- x;
  return (box(x1!), box(x2!), box(x3!));
}

/**
 * Box elements of a tuple.
 */
function box<Value1,Value2,Value3,Value4>(x:(Value1,Value2,Value3,Value4)) ->
    (Expression<Value1>, Expression<Value2>, Expression<Value3>, Expression<Value4>) {
  x1:Value1?;
  x2:Value2?;
  x3:Value3?;
  x4:Value4?;
  (x1, x2, x3, x4) <- x;
  return (box(x1!), box(x2!), box(x3!), box(x4!));
}

/**
 * Box elements of a tuple.
 */
function box<Value1,Value2,Value3,Value4,Value5>(x:(Value1,Value2,Value3,Value4,Value5)) ->
    (Expression<Value1>, Expression<Value2>, Expression<Value3>, Expression<Value4>, Expression<Value5>) {
  x1:Value1?;
  x2:Value2?;
  x3:Value3?;
  x4:Value4?;
  x5:Value5?;
  (x1, x2, x3, x4, x5) <- x;
  return (box(x1!), box(x2!), box(x3!), box(x4!), box(x5!));
}

/**
 * Box elements of a tuple.
 */
function box<Value1,Value2,Value3,Value4,Value5,Value6>(x:(Value1,Value2,Value3,Value4,Value5,Value6)) ->
    (Expression<Value1>, Expression<Value2>, Expression<Value3>, Expression<Value4>, Expression<Value5>, Expression<Value6>) {
  x1:Value1?;
  x2:Value2?;
  x3:Value3?;
  x4:Value4?;
  x5:Value5?;
  x6:Value6?;
  (x1, x2, x3, x4, x5, x6) <- x;
  return (box(x1!), box(x2!), box(x3!), box(x4!), box(x5!), box(x6!));
}

/**
 * Box elements of a tuple.
 */
function box<Value1,Value2,Value3,Value4,Value5,Value6,Value7>(x:(Value1,Value2,Value3,Value4,Value5,Value6,Value7)) ->
    (Expression<Value1>, Expression<Value2>, Expression<Value3>, Expression<Value4>, Expression<Value5>, Expression<Value6>, Expression<Value7>) {
  x1:Value1?;
  x2:Value2?;
  x3:Value3?;
  x4:Value4?;
  x5:Value5?;
  x6:Value6?;
  x7:Value7?;
  (x1, x2, x3, x4, x5, x6, x7) <- x;
  return (box(x1!), box(x2!), box(x3!), box(x4!), box(x5!), box(x6!), box(x7!));
}

/**
 * Box elements of a tuple.
 */
function box<Value1,Value2,Value3,Value4,Value5,Value6,Value7,Value8>(x:(Value1,Value2,Value3,Value4,Value5,Value6,Value7,Value8)) ->
    (Expression<Value1>, Expression<Value2>, Expression<Value3>, Expression<Value4>, Expression<Value5>, Expression<Value6>, Expression<Value7>, Expression<Value8>) {
  x1:Value1?;
  x2:Value2?;
  x3:Value3?;
  x4:Value4?;
  x5:Value5?;
  x6:Value6?;
  x7:Value7?;
  x8:Value8?;
  (x1, x2, x3, x4, x5, x6, x7, x8) <- x;
  return (box(x1!), box(x2!), box(x3!), box(x4!), box(x5!), box(x6!), box(x7!), box(x8!));
}
