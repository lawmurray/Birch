/**
 * Alive particle filter. When propagating and weighting particles, the
 * alive particle filter maintains $N$ particles with non-zero weight, rather
 * than $N$ particles in total as with the standard particle filter.
 */
class AliveParticleFilter < ParticleFilter {
  fiber filter(model:Model) -> (Model[_], Real[_], Real, Real, Integer) {
    auto x <- clone<Model>(model, nparticles);  // particles
    auto w <- vector(0.0, nparticles);  // log-weights
    auto ess <- 0.0;  // effective sample size
    auto S <- 0.0;  // logarithm of the sum of weights
    auto W <- 0.0;  // cumulative log normalizing constant estimate

    /* number of steps */
    if !nsteps? {
      nsteps <- model.size();
    }

    /* event handler */
    h:Handler <- play;
    if delayed {
      h <- global.delay;
    }

    /* initialize and weight */
    parallel for n in 1..nparticles {
      w[n] <- h.handle(x[n].simulate());
    }
    (ess, S) <- resample_reduce(w);
    W <- W + S - log(nparticles);
    yield (x, w, W, ess, nparticles);
   
    for t in 1..nsteps! {
      /* resample */
      auto a <- resample_systematic(w);
      dynamic parallel for n in 1..nparticles {
        if a[n] != n {
          x[n] <- clone<Model>(x[a[n]]);
        }
        w[n] <- 0.0;
      }

      /* propagate and weight */
      auto p <- vector(0, nparticles + 1);
      auto x0 <- x;
      auto w0 <- w;
      parallel for n in 1..nparticles + 1 {
        if n <= nparticles {
          x[n] <- clone<Model>(x0[a[n]]);
          w[n] <- h.handle(x[n].simulate(t));
          p[n] <- 1;
          while w[n] == -inf {  // repeat until weight is positive
            a[n] <- global.ancestor(w0);
            x[n] <- clone<Model>(x0[a[n]]);
            p[n] <- p[n] + 1;
            w[n] <- h.handle(x[n].simulate(t));
          }
        } else {
          /* propagate and weight until one further acceptance, which is
           * discarded for unbiasedness in the normalizing constant
           * estimate */
          auto w' <- 0.0;
          p[n] <- 0;
          do {
            auto a' <- global.ancestor(w0);
            auto x' <- clone<Model>(x0[a']);
            p[n] <- p[n] + 1;
            w' <- h.handle(x'.simulate(t));
          } while w' == -inf;  // repeat until weight is positive
        }
      }

      auto npropagations <- sum(p);
      (ess, S) <- resample_reduce(w);
      W <- W + S - log(npropagations - 1);
      yield (x, w, W, ess, npropagations);
    }
  }

  /**
   * Conditional filter.
   */
  fiber filter(model:Model, reference:Trace?) -> (Model[_], Real[_], Real,
      Real, Integer) {
    error("conditional filter not yet supported for AliveParticleFilter");
  }
}
