/**
 * Vector-borne disease model state.
 */
class VBDState {
  h:SEIRState;  // humans
  m:SEIRState;  // mosquitos

  function write(buffer:Buffer) {
    buffer.set("h", h);
    buffer.set("m", m);
  }
}
