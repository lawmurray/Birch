/*
 * Test the pmf of a univariate Boolean distribution.
 *
 * - π: The target distribution. 
 * - N: Number of samples.
 */
function test_pdf(π:Distribution<Boolean>, N:Integer) {  
  /* simulate, counting the occurrence of each value */
  auto k <- 0;
  for auto n in 1..N {
    if π.simulate() {
      k <- k + 1;
    }
  }

  /* compare pdf to count */
  auto failed <- false;
  auto ε <- 5.0/sqrt(N);
  
  auto δ <- abs(π.pdf(true) - Real(k)/N);
  if δ > ε {
    failed <- true;
    stderr.print("failed on true, " + δ + " > " + ε + "\n");
  }
  δ <- abs(π.pdf(false) - Real(N - k)/N);
  if δ > ε {
    failed <- true;
    stderr.print("failed on false, " + δ + " > " + ε + "\n");
  }
  if failed {
    exit(1);
  }
}

/*
 * Test the pmf of a univariate discrete distribution.
 *
 * - π: The target distribution. 
 * - N: Number of samples.
 */
function test_pdf(π:Distribution<Integer>, N:Integer) {  
  /* lower bound on interval */
  auto from <- π.lower();
  if !from? {
    from <- π.quantile(1.0e-6);
    assert from?;
  }

  /* upper bound on interval */
  auto to <- π.upper();
  if !to? {
    to <- π.quantile(1.0 - 1.0e-6);
    assert to?;
  }

  /* simulate, counting the occurrence of each value */
  auto count <- vector(0, to! - from! + 1);
  for auto n in 1..N {
    auto i <- π.simulate() - from! + 1;
    count[i] <- count[i] + 1;
  }

  /* compare sum of pdf to counts */
  auto failed <- false;
  for auto x in from!..to! {
    auto δ <- abs(π.pdf(x) - Real(count[x - from! + 1])/N);
    auto ε <- 5.0/sqrt(N);
    if δ > ε {
      failed <- true;
      stderr.print("failed on value " + x + ", " + δ + " > " + ε + "\n");
    }
  }
  if failed {
    exit(1);
  }
}

/*
 * Test a multivariate pdf.
 *
 * - π: The target distribution. 
 * - D: Number of dimensions.
 * - N: Number of samples.
 * - B: Number of samples in each batch for tuning Metropolis sampler.
 * - S: Metropolis sampler skip.
 */
function test_pdf(π:Distribution<Real[_]>, D:Integer, N:Integer, B:Integer,
    S:Integer) {  
  X1:Real[N,D];
  X2:Real[N,D];
  
  /* iid samples */
  for auto n in 1..N {
    X1[n,1..D] <- π.simulate();
  }
  
  /* tune the proposal distribution for a Metropolis sampler */
  auto σ2 <- vector(1.0, D);
  auto done <- false;
  do {
    auto a <- vector(0.0, D);  // acceptance rate for each component
    auto x <- π.simulate();

    for auto n in 1..B {
      for auto i in 1..D {
        auto x' <- x;
        x'[i] <- simulate_gaussian(x'[i], σ2[i]);
        if log(simulate_uniform(0.0, 1.0)) <= π.logpdf(x') - π.logpdf(x) {
          x <- x';  // accept
          a[i] <- a[i] + 1.0/B;
        }
      }
    }

    //stderr.print("\ntrial acceptance rates\n");
    //stderr.print("----------------------\n");
    //stderr.print(a + "\n");
    
    done <- true;
    for auto i in 1..D {
      if a[i] < 0.2 {
        done <- false;
        σ2[i] <- 0.5*σ2[i];
      } else if a[i] > 0.4 {
        done <- false;
        σ2[i] <- 1.5*σ2[i];
      }
    }
  } while !done;

  /* apply the Metropolis sampler */
  auto x <- π.simulate();
  auto a <- vector(0.0, D);  // acceptance rate for each component
  for auto n in 1..N {
    for auto s in 1..S {
      for auto i in 1..D {
        auto x' <- x;
        x'[i] <- simulate_gaussian(x'[i], σ2[i]);
        if log(simulate_uniform(0.0, 1.0)) <= π.logpdf(x') - π.logpdf(x) {
          x <- x';  // accept
          a[i] <- a[i] + 1.0/(N*S);
        }
      }
    }
    X2[n,1..D] <- vector(x);
  }

  //stderr.print("\nfinal acceptance rates\n");
  //stderr.print("----------------------\n");
  //stderr.print(a + "\n");
  
  /* test distance between the iid and Metropolis samples */
  if !pass(X1, X2) {
    exit(1);
  }
}

/*
 * Test a matrix pdf.
 *
 * - π: The target distribution. 
 * - R: Number of rows.
 * - C: Number of columns.
 * - N: Number of samples.
 * - B: Number of samples in each batch for tuning Metropolis sampler.
 * - S: Metropolis sampler skip.
 */
function test_pdf(π:Distribution<Real[_,_]>, R:Integer, C:Integer, N:Integer,
    B:Integer, S:Integer) {
  X1:Real[N,R*C];
  X2:Real[N,R*C];
  
  /* iid samples */
  for auto n in 1..N {
    X1[n,1..R*C] <- vector(π.simulate());
  }
  
  /* tune the proposal distribution for a Metropolis sampler */
  auto σ2 <- matrix(1.0, R, C);
  auto done <- false;
  do {
    auto a <- matrix(0.0, R, C);  // acceptance rate for each component
    auto x <- π.simulate();

    for auto n in 1..B {
      for auto i in 1..R {
        for auto j in 1..C {
          auto x' <- x;
          x'[i,j] <- simulate_gaussian(x'[i,j], σ2[i,j]);
          if log(simulate_uniform(0.0, 1.0)) <= π.logpdf(x') - π.logpdf(x) {
            x <- x';  // accept
            a[i,j] <- a[i,j] + 1.0/B;
          }
        }
      }
    }

    //stderr.print("\ntrial acceptance rates\n");
    //stderr.print("----------------------\n");
    //stderr.print(a + "\n");
    
    done <- true;
    for auto i in 1..R {
      for auto j in 1..C {
        if a[i,j] < 0.2 {
          done <- false;
          σ2[i,j] <- 0.5*σ2[i,j];
        } else if a[i,j] > 0.4 {
          done <- false;
          σ2[i,j] <- 1.5*σ2[i,j];
        }
      }
    }
  } while !done;

  /* apply the Metropolis sampler */
  auto x <- π.simulate();
  auto a <- matrix(0.0, R, C);  // acceptance rate for each component
  for auto n in 1..N {
    for auto s in 1..S {
      for auto i in 1..R {
        for auto j in 1..C {
          auto x' <- x;
          x'[i,j] <- simulate_gaussian(x'[i,j], σ2[i,j]);
          if log(simulate_uniform(0.0, 1.0)) <= π.logpdf(x') - π.logpdf(x) {
            x <- x';  // accept
            a[i,j] <- a[i,j] + 1.0/(N*S);
          }
        }
      }
    }
    X2[n,1..R*C] <- vector(x);
  }

  //stderr.print("\nfinal acceptance rates\n");
  //stderr.print("----------------------\n");
  //stderr.print(a + "\n");
  
  /* test distance between the iid and Metropolis samples */
  if !pass(X1, X2) {
    exit(1);
  }
}
