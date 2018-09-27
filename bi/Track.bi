class Track < StateSpaceModel<Global,Random<Real[_]>,Random<Real[_]>> {
  /**
   * Starting time of this track.
   */
  t:Integer;

  fiber initial(x':Random<Real[_]>, θ:Global) -> Real {
    auto μ <- vector(0.0, 3*length(θ.l));
    μ[1..2] <~ Uniform(θ.l, θ.u);
    x' ~ Gaussian(μ, θ.M);
  }
  
  fiber transition(x':Random<Real[_]>, x:Random<Real[_]>, θ:Global) -> Real {
    x' ~ Gaussian(θ.A*x, θ.Q);
  }

  fiber observation(y':Random<Real[_]>, x:Random<Real[_]>, θ:Global) -> Real {
    y' ~ Gaussian(θ.B*x, θ.R);
  }

  function read(reader:Reader) {
    //
  }
  
  function write(writer:Writer) {
    writer.set("t", t);
    x.write(writer.setObject("x"));
    y.write(writer.setObject("y"));
  }
}
