/**
 * Model for Yap case study.
 */
class YapDengueModel < MarkovModel<YapDengueParameter,YapDengueState> {
  v:VBDModel;

  fiber parameter(θ:YapDengueParameter) -> Event {
    θ.h.ν <- 0.0;
    θ.h.μ <- 1.0;
    θ.h.λ ~ Beta(1.0, 1.0);
    θ.h.δ ~ Beta(1.0 + 2.0/4.4, 3.0 - 2.0/4.4);
    θ.h.γ ~ Beta(1.0 + 2.0/4.5, 3.0 - 2.0/4.5);

    θ.m.ν <- 1.0/7.0;
    θ.m.μ <- 6.0/7.0;
    θ.m.λ ~ Beta(1.0, 1.0);
    θ.m.δ ~ Beta(1.0 + 2.0/6.5, 3.0 - 2.0/6.5);
    θ.m.γ <- 0.0;

    θ.ρ ~ Beta(1.0, 1.0);
  }

  fiber initial(x:YapDengueState, θ:YapDengueParameter) -> Event {
    x.h.n <- 7370;
    x.h.i <- 1 + simulate_poisson(5.0);
    x.h.e <- simulate_poisson(5.0);
    x.h.r <- Integer(simulate_uniform(0.0, x.h.n - x.h.i - x.h.e));
    x.h.s <- x.h.n - x.h.e - x.h.i - x.h.r;

    x.h.Δs <- 0;
    x.h.Δe <- x.h.e;
    x.h.Δi <- x.h.i;
    x.h.Δr <- 0;
    
    x.m.n <- Integer(x.h.n*pow(10.0, simulate_uniform(-1.0, 2.0)));
    x.m.r <- 0;
    x.m.i <- 0;
    x.m.e <- 0;
    x.m.s <- x.m.n;

    x.m.Δs <- 0;
    x.m.Δe <- 0;
    x.m.Δi <- 0;
    x.m.Δr <- 0;

    x.z <- x.h.Δi;
    if (x.y?) {
      x.y! ~> Binomial(x.z, θ.ρ);
      x.z <- 0;
    }
  }
  
  fiber transition(x':YapDengueState, x:YapDengueState, θ:YapDengueParameter) -> Event {
    v.transition(x', x, θ);
    x'.z <- x.z + x'.h.Δi;
    if (x'.y?) {
      x'.y! ~> Binomial(x'.z, θ.ρ);
      x'.z <- 0;
    }
  }
}
