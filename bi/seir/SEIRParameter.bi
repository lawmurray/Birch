/**
 * Parameters of an SEIR model.
 */
class SEIRParameter {
  ν:Random<Real>;   // birth probability
  μ:Random<Real>;   // survival probability
  λ:Random<Real>;   // exposure probability
  δ:Random<Real>;   // infection probability
  γ:Random<Real>;   // recovery probability
  
  fiber run() -> Real! {
    ν <- 0.0;
    μ <- 1.0;
    λ ~ Beta(1.0, 1.0);
    δ ~ Beta(1.0, 1.0);
    γ ~ Beta(1.0, 1.0);
  }

  function output(prefix:String) {    
    output(prefix, "ν", ν);
    output(prefix, "μ", μ);
    output(prefix, "λ", λ);
    output(prefix, "δ", δ);
    output(prefix, "γ", γ);
  }
  
  function output(prefix:String, name:String, value:Real) {
    out:FileOutputStream;
    out.open(prefix + name + ".csv", "a");
    out.print(value + "\n");
    out.close();
  }
}
