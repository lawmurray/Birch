/**
 * Vector-borne disease model parameters.
 */
class VBDParameter {
  h:SEIRParameter;  // humans
  m:SEIRParameter;  // mosquitos

  function write(buffer:Buffer) {
    buffer.set("h", h);
    buffer.set("m", m);
  }
}
