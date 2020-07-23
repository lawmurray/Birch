/**
 * Parameter for PoissonGaussianModel.
 */
class PoissonGaussianParameter {
  a:Real <- 0.8;
  b:Real <- 10.0;
  σ2_x:Real <- 1.0;
  σ2_y:Real <- 0.01;

  function read(buffer:Buffer) {
    a <-? buffer.get("a", a);
    b <-? buffer.get("b", b);
    σ2_x <-? buffer.get("σ2_x", σ2_x);
    σ2_y <-? buffer.get("σ2_y", σ2_y);
  }

  function write(buffer:Buffer) {
    buffer.set("a", a);
    buffer.set("b", b);
    buffer.set("σ2_x", σ2_x);
    buffer.set("σ2_y", σ2_y);
  }
}

/**
 * State-space model with linear-Gaussian dynamicss and Poisson observation.
 */
class PoissonGaussianModel < StateSpaceModel<PoissonGaussianParameter,
    Random<Real>,Random<Integer>> {
  fiber initial(x:Random<Real>, θ:PoissonGaussianParameter) -> Event {
    x ~ Gaussian(0.0, θ.σ2_x);
  }

  fiber transition(x':Random<Real>, x:Random<Real>,
      θ:PoissonGaussianParameter) -> Event {
    x' ~ Gaussian(θ.a*x, θ.σ2_x);
  }

  fiber observation(y:Random<Integer>, x:Random<Real>,
      θ:PoissonGaussianParameter) -> Event {
    z:Random<Real>;
    z ~ Gaussian(θ.b*x, θ.σ2_y);
    y ~ Poisson(exp(0.1*z));
  }
}
