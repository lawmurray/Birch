/**
 * Record of a delayed value.
 *
 * - v: The random variate that will contain the delayed value.
 */
final class DelayRecord<Value>(v:Random<Value>) < ValueRecord<Value> {
  /**
   * Random variate that will contain the delayed value.
   */
  v:Random<Value> <- v;

  function hasValue() -> Boolean {
    return v.hasValue();
  }

  function value() -> Value {
    return v.value();
  }
  
  function random() -> Random<Value> {
    return v;
  }

  function ratio(record:Record) -> Real {
    auto current <- DelayRecord<Value>?(record);
    if current? {
      auto v' <- this.v;
      auto v <- current!.v;
      auto α <- v'.w - v.w;
      if v.x? && v.dfdx? && v'.x? && v'.dfdx? {
        α <- α + logpdf_propose(v.x!, v'.x!, v'.dfdx!);
        α <- α - logpdf_propose(v'.x!, v.x!, v.dfdx!);
      }
      return α;
    } else {
      return 0.0;
    }
  }
}

/**
 * Create a DelayRecord.
 */
function DelayRecord<Value>(v:Random<Value>) -> DelayRecord<Value> {
  evt:DelayRecord<Value>(v);
  return evt;
}
