/**
 * Parameter model for Yap case study.
 */
class YapParameter < VBDParameter {
  ρ:Random<Real>;  // probability of an actual case being observed

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("ρ", ρ);
  }  
}
