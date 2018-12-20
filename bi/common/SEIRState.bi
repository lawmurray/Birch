/**
 * SEIR process model.
 */
class SEIRState {
  Δs:Integer;  // newly susceptible (births)
  Δe:Integer;  // newly exposed
  Δi:Integer;  // newly infected
  Δr:Integer;  // newly recovered

  s:Integer;   // susceptible population
  e:Integer;   // incubating population
  i:Integer;   // infectious population
  r:Integer;   // recovered population

  n:Integer;   // total population
  
  function write(buffer:Buffer) {
    buffer.set("n", n);
    buffer.set("s", s);
    buffer.set("e", e);
    buffer.set("i", i);
    buffer.set("Δs", Δs);
    buffer.set("Δe", Δe);
    buffer.set("Δi", Δi);
    buffer.set("Δr", Δr);
  }
}
