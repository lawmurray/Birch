/**
 * Event handler that replays a trace of events in delayed mode.
 *
 * - replay: The trqce to replay. This may be either an immediate trace or a
 *   delayed trace. For the former, the outcome is the same as using
 *   ImmediateReplayHandler.
 */
class DelayedReplayHandler < ReplayHandler {
  function handle(evt:FactorEvent) -> Real {
    return evt.observe();
  }
  
  function handle(evt:RandomEvent) -> Real {
    auto replayEvt <- next();
    if evt.hasValue() {
      ///@todo Check that values match
      return evt.observe();
    } else {
      evt.assume();
      if replayEvt? && replayEvt!.hasValue() {
        /* still delay sampling, but register the required value for when the
         * variate is ultimately simulated */
        evt.value(replayEvt!);
      }
      return 0.0;
    }
  }
}
