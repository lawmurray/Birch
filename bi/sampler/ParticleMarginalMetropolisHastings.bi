class ParticleMarginalMetropolisHastings < ParticleFilter {
  /* This sampler is a first step to getting a Metropolis-Hastings step within
   * the particle filter. The code should propose parameter values before running
   * the particle filter, then accept the parameters with probability given by
    * 
    *  p(y|θ')p(θ')q(θ|θ')
    *  -------------------
    *   p(y|θ)p(θ)q(θ'|θ)
    *
    * Where `θ` is the previous parameter value and `θ'` is the proposed parameter value.
    *
    * The code works and samples something; however, there is some problem and the samples
    * drawn do not come from the correct posterior distribution. I do not know if it's
    * something in the way I compute the log evidence, if it is something in the cloning
    * steps (something you hinted at in the meeting), or if it is something in my `propose`
    * function.
    */
  m:ForwardModel?; // previous model
  py:Real; // evidence of the previous model
  q:Real; // proposal weight of the previous model
  m':ForwardModel; // proposed model
  py':Real; // evidence of the proposed model
  q':Real; // proposal weight of the proposed model
  A:Real <- 0; // accepted samples
  R:Real <- 0; // rejected samples


  function initialize() {
    if !m? {
      /* No pervious model, run as normal. */
      super.initialize();
    } else {
      /* There is a previous model, create particles from  proposal model 
       * The idea is to use PMMH to sample the parameters that cannot be integrated out.
       * To simplify the matter, we can assume that these parameters are all in the `parameters` fiber
       * and that any parameters that can be marginalized are in the state.
       */
      
      /* We clone a fresh empty model (no random variables set), from the archetype to create our proposal. */
      m' <- clone<ForwardModel>(archetype!); // Create a proposal model from the archetype

      /* The propose member function takes a model as an argument and assigns parameters based on a certain proposal.
       * In addition, it returns `q' = log p(θ' | θ)` and `q = log p(θ | θ')`
       */
      (q', q) <- m'.propose(m!); // Set the parameters of the proposed model from the previous model

      /* We then proceed as the normal particle filter but we copy the propsed model to all particles */
      Z.clear();
      ess.clear();
      elapsed.clear();

      w <- vector(0.0, N);
      a <- iota(1, N);
      o <- vector(1, N);

      /* Clone the proposed model to all particles */
      x1:Vector<ForwardModel>;
      x1.enlarge(N, clone<ForwardModel>(m'));
      x <- x1.toArray();

      for auto n in 1..N {
        x[n] <- clone<ForwardModel>(x[n]);
      }

      tic();
    }
  }
  
  function finalize() {
    /* After the `sample()` function, the filter has drawn x[b] and has collected the associated log evidence `Z`
     * Because we have assigned the prior parameters, `Z` should contain `log p(y | θ') + log p(θ')` -- that is, the 
     * evidence of the proposed model
     */
    m' <- x[b];
    py' <- sum(Z.walk());
    if m? {
      /* The actual marginal Metropolis-Hastings step, where we accept with probability
       * 
       *  p(y|θ')p(θ')q(θ|θ')
       *  -------------------
       *   p(y|θ)p(θ)q(θ'|θ)
       */  
      if (log(simulate_uniform(0.0, 1.0)) < py' + q - py - q') {
        /* Accept the model and save it into m (accepted model used @34 to propose the parameters) */
        m <- clone<ForwardModel>(m');
        py <- py'; // Save the new evidence
        A <- A + 1;
      } else {
        /* Reject the model and replace the particle with the previous model.
         * Ideally, we should also overwrite `Z`, but the value is not used.
         */
        x[b] <- clone<ForwardModel>(m!);
        R <- R + 1;
      }
    } else {
      /* we do not have a previous model, just accept it */
      m <- clone<ForwardModel>(m');
      py <- py';
    }
    if verbose {
      stderr.print("acceptance rate: " + A/(A+R) + "\n");
    }
  }

}