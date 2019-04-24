/**
 * Particle marginal Metropolis--Hastings sampler.
 *
 * !!! caution
 *     Work in progress, contributions welcome.
 */
class ParticleMarginalMetropolisHastings < Sampler {
  /**
   * Inner sampler.
   */
  sampler:Sampler?;
  
  /**
   * Current state.
   */
  x:Model?;
  
  /**
   * Evaluation of target at current state, $\pi(x)$.
   */
  π:Real?;
    
  function sample(m:Model) -> (Model, Real) {
    /* Markov chain states and evaluations */
    x':Model <- m;
    π':Real;  // π(x')
    q:Real;   // q(x'|x)
    q':Real;  // q(x|x')
    
    if !x? {
      /* initial state */
      (x, π) <- sampler!.sample();
    } else {
      /* subsequent states */
      x' <- clone<Model>(m);
      
      /* simulate a proposed state */
      //q' <- sum(limit(x'.propose(x!), 1));
      
      /* re-execute to now compute the proposal log-weight, subtracting out
       * the previous log-weight as it represents log-likelihood terms that
       * are in common */
      //q' <- sum(limit(x'.propose(x!), 1));
      
      /* execute the other way to compute the reverse proposal log-weight */
      //q <- sum(limit(x!.propose(x'), 1));
      
      /* compute the target density, π' will be the sum of the log-likelihood
       * and log-prior */
      (x', π') <- sampler!.sample();
      
      /* accept with probability given by Metropolis--Hastings rule */
      auto α <- exp(min(π' - π! + q - q', 0.0));
      if simulate_bernoulli(α) {
        (x, π) <- (x', π');  // accept
      }
    }
    return (x!, 0.0);  // return with zero log-weight as MCMC here
  }

  function read(buffer:Buffer) {
    super.read(buffer);

    /* create the inner sampler */
    className:String?;
    samplerBuffer:Buffer? <- buffer.getObject("sampler");
    if (samplerBuffer?) {
      className <- samplerBuffer!.getString("class");
    }
    if (!className?) {
      className <- "ParticleFilter";
    }
    sampler <- Sampler?(make(className!));
    if (!sampler?) {
      error(className! + " is not a subtype of Sampler.");
    }
    buffer.get("sampler", sampler!);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("sampler", sampler!);
  }
}
