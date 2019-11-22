/**
 * Alive particle filter. When propagating and weighting particles, the
 * alive particle filter maintains $N$ particles with non-zero weight, rather
 * than $N$ particles in total as with the standard particle filter.
 */
class AliveParticleFilter {
  /**
   * Number of particles.
   */
  nparticles:Integer <- 1;

  /**
   * Should delayed sampling be used?
   */
  delayed:Boolean <- true;

  fiber filter(model:Model) -> (Model[_], Real[_], Real, Real, Integer) {
    auto x <- clone<Model>(model, nparticles);  // particles
    auto w <- vector(0.0, 0);  // log-weights
    auto V <- 0.0;  // incrmental log normalizing constant estimate
    auto W <- 0.0;  // cumulative log normalizing constant estimate
    auto ess <- 0.0;  // effective sample size
    
    /* event handler */
    h:Handler <- play;
    if delayed {
      h <- global.delay;
    }

    /* initialize and weight */
    parallel for n in 1..nparticles {
      w[n] <- h.handle(x[n].simulate());
    }
    (ess, V) <- resample_reduce(w);
    W <- W + V;
    yield (x, w, W, ess, nparticles);
   
    auto t <- 0;
    while true {
      t <- t + 1;

      /* resample */
      auto a <- resample_systematic(w);
      dynamic parallel for n in 1..nparticles {
        if a[n] != n {
          x[n] <- clone<Model>(x[a[n]]);
        }
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
            a[n] <- ancestor(w0);
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
            auto a' <- ancestor(w0);
            auto x' <- clone<Model>(x0[a']);
            p[n] <- p[n] + 1;
            w' <- h.handle(x'.simulate(t));
          } while w' == -inf;  // repeat until weight is positive
        }
      }

      (ess, V) <- resample_reduce(w);
      auto npropagations <- sum(p);
      V <- V + log(nparticles) - log(npropagations - 1);
      W <- W + V;
    yield (x, w, W, ess, npropagations);
    }
  }

  function read(buffer:Buffer) {
    nparticles <-? buffer.get("nparticles", nparticles);
    delayed <-? buffer.get("delayed", delayed);
  }

  function write(buffer:Buffer) {
    buffer.set("nparticles", nparticles);
    buffer.set("delayed", delayed);
  }
}
