/**
 * SEIR process model.
 */
final class SEIRState {
  Δs:Integer <- 0;  // newly susceptible (births)
  Δe:Integer <- 0;  // newly exposed
  Δi:Integer <- 0;  // newly infected
  Δr:Integer <- 0;  // newly recovered

  s:Integer <- 0;   // susceptible population
  e:Integer <- 0;   // incubating population
  i:Integer <- 0;   // infectious population
  r:Integer <- 0;   // recovered population

  n:Integer <- 0;   // total population
  
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
