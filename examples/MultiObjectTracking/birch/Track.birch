class Track < StateSpaceModel<Global,Random<Real[_]>,Random<Real[_]>> {
  /**
   * Starting time of this track.
   */
  t:Integer;

  fiber initial(x:Random<Real[_]>, θ:Global) -> Event {
    auto μ <- vector(0.0, 3*length(θ.l));
    μ[1..2] <~ Uniform(θ.l, θ.u);
    x ~ Gaussian(μ, θ.M);
  }
  
  fiber transition(x':Random<Real[_]>, x:Random<Real[_]>, θ:Global) -> Event {
    x' ~ Gaussian(θ.A*x, θ.Q);
  }

  fiber observation(y:Random<Real[_]>, x:Random<Real[_]>, θ:Global) -> Event {
    d:Boolean;
    d <~ Bernoulli(θ.d);  // is the track detected?
    if d {
      y ~ Gaussian(θ.B*x, θ.R);
    }
  }
  
  function write(buffer:Buffer) {
    /* previous versions of the Birch standard library automatically
     * triggered simulation of Random objects on output; newer versions do
     * not, to force this */
    auto f <- x.walk();
    while f? {
      f!.value();
    }
    auto g <- y.walk();
    while g? {
      if g!.hasDistribution() {
        g!.value();
      }
    }
    buffer.set("t", t);
    super.write(buffer);
  }
}
