/**
 * State model for Yap case study.
 */
class YapState < VBDState {
  z:Integer;   // actual number of new cases since last observation
  y:Integer?;  // observed number of cases since last observation
  
  function read(buffer:Buffer) {
    y <- buffer.getInteger();
  }
}
