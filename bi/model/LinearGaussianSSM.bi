class LinearGaussianSSMParameter < Model {
  a:Real <- 0.8;
  σ2_x:Real <- 1.0;
  σ2_y:Real <- 0.1;
}

class LinearGaussianSSMState < State {
  x:Random<Real>;
  y:Random<Real>;

  fiber simulate(θ:LinearGaussianSSMParameter) -> Real! {
    x <~ Gaussian(0.0, θ.σ2_x);
    y ~ Gaussian(x, θ.σ2_y);
  }

  fiber simulate(χ:LinearGaussianSSMState, θ:LinearGaussianSSMParameter) -> Real! {
    x <~ Gaussian(θ.a*χ.x, θ.σ2_x);
    y ~ Gaussian(x, θ.σ2_y);
  }

  function input(reader:Reader) {
    y <- reader.getReal();
  }

  function output(writer:Writer) {
    writer.setReal(x);
  }
}

class LinearGaussianSSM = MarkovModel<LinearGaussianSSMState,LinearGaussianSSMParameter>;
