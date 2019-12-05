/**
 * Parameter for PoissonGaussianModel.
 */
class PoissonGaussianParameter = LinearGaussianParameter;
 
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
    y ~ Poisson(exp(z));
  }
}
