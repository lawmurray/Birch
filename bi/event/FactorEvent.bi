/**
 * Event triggered for a factor.
 *
 * - w: Associated log-weight.
 *
 * The Event class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Event.svg"></object>
 * </center>
 */
final class FactorEvent(w:Real) < Event {
  /**
   * Associated (log-)weight.
   */
  w:Real <- w;
  
  function record() -> Record {
    return FactorRecord(w);
  }

  function coerce(record:Record) -> FactorRecord {
    auto r <- FactorRecord?(record);
    if !r? {
      error("incompatible trace");
    }
    return r!;
  }

  function accept(handler:PlayHandler) -> Real {
    return handler.handle(this);
  }
  
  function accept(handler:MoveHandler) -> Real {
    return handler.handle(this);
  }

  function accept(record:Record, handler:PlayHandler) -> Real {
    return handler.handle(coerce(record), this);
  }

  function accept(record:Record, handler:MoveHandler) -> Real {
    return handler.handle(coerce(record), this);
  }
}

/**
 * Create a FactorEvent.
 */
function FactorEvent(w:Real) -> FactorEvent {
  evt:FactorEvent(w);
  return evt;
}
