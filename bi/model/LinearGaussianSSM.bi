/**
 * Linear-Gaussian state-space model.
 */
class LinearGaussianSSM = MarkovModel<LinearGaussianSSMState,
    LinearGaussianSSMParameter>;

/**
 * Linear-Gaussian state-space model parameter.
 */
class LinearGaussianSSMParameter < Parameter {
  a:Real <- 0.8;
  σ2_x:Real <- 1.0;
  σ2_y:Real <- 0.1;
}

/**
 * Linear-Gaussian state-space model state.
 */
class LinearGaussianSSMState < State {
  x:Random<Real>;
  y:Random<Real>;

  fiber initial(θ:LinearGaussianSSMParameter) -> Real {
    x ~ Gaussian(0.0, θ.σ2_x);
    y ~ Gaussian(x, θ.σ2_y);
  }

  fiber transition(χ:LinearGaussianSSMState, θ:LinearGaussianSSMParameter) -> Real {
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
