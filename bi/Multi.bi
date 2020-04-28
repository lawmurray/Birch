class Multi < StateSpaceModel<Global,Vector<Track>,Vector<Random<Real[_]>>> {
  /**
   * Current time.
   */
  t:Integer <- 0;

  /**
   * All tracks up to current time.
   */
  z:Vector<Track>;
  
  fiber transition(x':Vector<Track>, x:Vector<Track>, θ:Global) -> Event {
    /* update time */
    t <- t + 1;

    /* move current objects */
    auto track <- x.walk();
    while track? {
      auto ρ <- exp(logpdf_poisson(t - track!.t - 1, θ.τ));
      auto R <- 1.0 - cdf_poisson(t - track!.t - 1, θ.τ) + ρ;
      s:Boolean;
      s <~ Bernoulli(1.0 - ρ/R);  // does the object survive?
      if s {
        track!.simulate(t - track!.t + 1)!!;
        x'.pushBack(track!);
      }
    }
    
    /* birth new objects */
    N:Integer;
    N <~ Poisson(θ.λ);
    for n in 1..N {
      track:Track;
      track.t <- t;
      track.θ <- θ;
      track.simulate()!!;   // up to parameters
      track.simulate(1)!!;  // up to initial time
      x'.pushBack(track);
      z.pushBack(track);
    }
  }

  fiber observation(y:Vector<Random<Real[_]>>, x:Vector<Track>, θ:Global) -> Event {
    if !y.empty() {
      association(y, x, θ)!!;
    } else {
      /* clutter */
      N:Integer;
      N <~ Poisson(θ.μ);
      for n in 1..(N + 1) {
        clutter:Random<Real[_]>;
        clutter <~ Uniform(θ.l, θ.u);
        y.pushBack(clutter);
      }
    }
  }

  fiber association(y:Vector<Random<Real[_]>>, x:Vector<Track>, θ:Global) -> Event {
    auto track <- x.walk();
    while track? {
      auto o <- track!.y.back();  // observed random variable
      if o.hasDistribution() {
        auto p <- o.distribution()!;
      
        /* object is detected, compute proposal */
        q:Real[y.size()];
        auto n <- 1;
        auto detection <- y.walk();
        while detection? {
          q[n] <- p.pdf(detection!.value());
          n <- n + 1;
        }
        auto Q <- sum(q);
                  
        /* propose an association */
        if Q > 0.0 {
          q <- q/Q;
          n <~ Categorical(q);  // propose an observation to associate with
          auto w <- p.observe(y.get(n).value());  // likelihood
          w <- w - log(Real(y.size()));  // prior correction (uniform prior)
          w <- w - log(q[n]);  // proposal correction
          y.erase(n);  // remove the observation for future associations
          yield FactorEvent(w);
        } else {
          yield FactorEvent(-inf);  // detected, but all likelihoods (numerically) zero
        }
      }
    }
    
    /* clutter */
    y.size() - 1 ~> Poisson(θ.μ);
    auto clutter <- y.walk();
    while clutter? {
      clutter!.value() ~> Uniform(θ.l, θ.u);
    }
  }
    
  function write(buffer:Buffer) {
    buffer.set("θ", θ);
    buffer.set("z", z);
    buffer.set("y", y);
  }
}
