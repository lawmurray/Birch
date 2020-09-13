/**
 * Parameters of an SEIR model.
 */
final class SEIRParameter {
  ν:Random<Real>;   // birth probability
  μ:Random<Real>;   // survival probability
  λ:Random<Real>;   // exposure probability
  δ:Random<Real>;   // infection probability
  γ:Random<Real>;   // recovery probability

  function write(buffer:Buffer) {
    buffer.set("ν", ν);
    buffer.set("μ", μ);
    buffer.set("λ", λ);
    buffer.set("δ", δ);
    buffer.set("γ", γ);
  }
}
