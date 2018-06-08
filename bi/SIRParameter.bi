/**
 * Parameter for SIRModel.
 */
class SIRParameter < Parameter {
  /**
   * Interaction rate.
   */
  λ:Random<Real>;

  /**
   * Infection probability.
   */
  δ:Random<Real>;

  /**
   * Recovery probability.
   */
  γ:Random<Real>;

  fiber parameter() -> Real {
    λ ~ Gamma(2.0, 5.0);
    δ ~ Beta(2.0, 2.0);
    γ ~ Beta(2.0, 2.0);
  }

  function input(reader:Reader) {
    λ <- reader.getReal("λ");
    δ <- reader.getReal("δ");
    γ <- reader.getReal("γ");
  }

  function output(writer:Writer) {
    writer.setReal("λ", λ);
    writer.setReal("δ", δ);
    writer.setReal("γ", γ);
  }
}
