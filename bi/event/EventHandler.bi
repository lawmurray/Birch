PLAY_IMMEDIATE:Integer8 <- Integer8(0);
PLAY_DELAY:Integer8 <- Integer8(1);
SKIP_IMMEDIATE:Integer8 <- Integer8(2);
SKIP_DELAY:Integer8 <- Integer8(3);
REPLAY_IMMEDIATE:Integer8 <- Integer8(4);
REPLAY_DELAY:Integer8 <- Integer8(5);
DOWNDATE_IMMEDIATE:Integer8 <- Integer8(6);
DOWNDATE_DELAY:Integer8 <- Integer8(7);
PROPOSE_IMMEDIATE:Integer8 <- Integer8(8);
NUM_MODES:Integer8 <- Integer8(9);

/**
 * Event handler.
 */
final class EventHandler {
  /**
   * Trace.
   */
  trace:Queue<Record>;
    
  /**
   * Mode.
   */
  auto mode <- PLAY_DELAY;

  /**
   * Is recording enabled?
   */
  auto record <- false;
    
  /**
   * Set the play/replay mode. Valid modes are:
   */
  function setMode(mode:Integer8) {
    assert 0 <= mode && mode <= NUM_MODES;
    this.mode <- mode;
  }

  /**
   * Set the record flag.
   */
  function setRecord(record:Boolean) {
    this.record <- record;
  }
  
  /**
   * Handle a sequence of events.
   *
   * Returns: Log-weight.
   */
  function handle(evt:Event!) -> Real {
    auto w <- 0.0;
    while w > -inf && evt? {
      w <- w + handle(evt!);
    }
    return w;
  }

  /**
   * Handle a single event.
   *
   * - evt: The event.
   *
   * Returns: Any necessary log-weight adjustment.
   */  
  function handle(evt:Event) -> Real {
    auto w <- 0.0;
    if mode == PLAY_IMMEDIATE {
      w <- evt.playImmediate();
    } else if mode == PLAY_DELAY {
      w <- evt.playDelay();
    } else if mode == SKIP_IMMEDIATE {
      w <- evt.skipImmediate(trace);
    } else if mode == SKIP_DELAY {
      w <- evt.skipDelay(trace);
    } else if mode == REPLAY_IMMEDIATE {
      w <- evt.replayImmediate(trace);
    } else if mode == REPLAY_DELAY {
      w <- evt.replayDelay(trace);
    } else if mode == DOWNDATE_IMMEDIATE {
      w <- evt.downdateImmediate(trace);
    } else if mode == DOWNDATE_DELAY {
      w <- evt.downdateDelay(trace);
    } else if mode == PROPOSE_IMMEDIATE {
      w <- evt.proposeImmediate(trace);
    } else {
      assert false;
    }
    if record {
      evt.record(trace);
    }
    return w;
  }
  
  /**
   * Rewind, in order to replay recorded events.
   */
  function rewind() {
    trace.allForward();
  }
}
