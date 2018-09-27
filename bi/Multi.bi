class Multi < StateSpaceModel<Global,List<Track>,List<Detection>> {
  /**
   * All tracks.
   */
  z:List<Track>;
  
  /**
   * Time.
   */
  t:Integer <- 1;

  fiber transition(x':List<Track>, x:List<Track>, θ:Global) -> Real {
    /* move current objects */
    auto track <- x.walk();
    while track? {
      s:Boolean;
      s <~ Bernoulli(θ.s);
      if (s) {
        track!.step();
        x'.pushBack(track!);
      }
    }
    
    /* birth new objects */
    N:Integer;
    N <~ Poisson(θ.λ);
    for n:Integer in 1..N {
      track:Track;
      track.t <- t;
      track.θ <- θ;
      track.start();
      x'.pushBack(track);
      z.pushBack(track);
    }
    
    t <- t + 1;
  }

  fiber observation(y':List<Detection>, x:List<Track>, θ:Global) -> Real {
    /* an empty list of observations is interpreted as missing observations,
     * and so they will be generated, otherwise data association to tracks is
     * used */
    associate:Boolean <- !y'.empty();
    n:Integer <- 0;
  
    /* current objects */
    auto track <- x.walk();
    while track? {
      d:Boolean;
      d <~ Bernoulli(θ.d);
      if d {
        /* object is detected */
        if associate {
          /* associate this object with an observation */
          p:Real[y'.size()];
          auto detection <- y'.walk();
          n <- 1;
          while detection? {
            p[n] <- track!.y.back().pdf(detection!);
            n <- n + 1;
          }
          P:Real <- sum(p);
          if P > 0.0 {
            p <- p/P;
            n <~ Categorical(p);
            yield track!.y.back().realize(y'.get(n)) - log(p[n]);
            y'.erase(n);
          } else {
            yield -inf;
          }
        }
      }
    }

    /* clutter */
    if associate {
      y'.size() - 1 ~> Poisson(θ.μ);
      auto detection <- y'.walk();
      while detection? {
        detection! ~> Uniform(θ.l, θ.u);
      }
    } else {
      N:Integer;
      N <~ Poisson(θ.μ);
      for n:Integer in 1..(N + 1) {
        detection:Detection;
        detection <~ Uniform(θ.l, θ.u);
        y'.pushBack(detection);
      }
    }
  }
  
  function read(reader:Reader) {
    θ.read(reader.getObject("θ"));
    y1.read(reader.getObject("y"));
  }
  
  function write(writer:Writer) {
    θ.write(writer.setObject("θ"));
    z.write(writer.setObject("z"));
    y.write(writer.setObject("y"));
  }
}
