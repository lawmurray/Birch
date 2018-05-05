/**
 * State model for linear-Gaussian state-space model.
 */
class LinearGaussianState < State {
  x:Random<Real>;
  y:Random<Real>;

  fiber initial(θ:LinearGaussianParameter) -> Real {
    x ~ Gaussian(0.0, θ.σ2_x);
    y ~ Gaussian(x, θ.σ2_y);
  }

  fiber transition(χ:LinearGaussianState, θ:LinearGaussianParameter) -> Real {
    x ~ Gaussian(θ.a*χ.x, θ.σ2_x);
    y ~ Gaussian(x, θ.σ2_y);
  }

  function input(reader:Reader) {
    y <- reader.getReal();
  }

  function output(writer:Writer) {
    writer.setReal(x);
  }
}
