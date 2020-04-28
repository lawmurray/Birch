/**
 * Record of an AssumeEvent.
 *
 * - x: Random variate.
 */
final class AssumeRecord<Value>(x:Random<Value>) < Record {
  /**
   * Random variate.
   */
  x:Random<Value> <- x;

  function ratio(record:Record, scale:Real) -> Real {
    auto current <- AssumeRecord<Value>?(record);
    auto x' <- this.x;
    auto x <- current!.x;
    auto α <- x'.w - x.w;
    if x.x? && x.dfdx? && x'.x? && x'.dfdx? {
      α <- α + logpdf_propose(x.x!, x'.x!, x'.dfdx!, scale);
      α <- α - logpdf_propose(x'.x!, x.x!, x.dfdx!, scale);
    }
    return α;
  }
}

/**
 * Create a AssumeRecord.
 */
function AssumeRecord<Value>(v:Random<Value>) -> AssumeRecord<Value> {
  evt:AssumeRecord<Value>(v);
  return evt;
}
