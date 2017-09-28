/**
 * Parameters for Yap case study.
 */
class YapParameter < VBDParameter {
  ρ:Beta;  // reporting probability
  
  fiber run() -> Real! {
    super.run();
    ρ ~ Beta(1.0, 1.0);
  }
}
