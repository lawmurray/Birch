/*
 * Test matrix normal-inverse-gamma pdf evaluations.
 */
program test_pdf_matrix_normal_inverse_gamma(R:Integer <- 3, C:Integer <- 2,
    N:Integer <- 10000, B:Integer <- 500, S:Integer <- 30) {
  M:Real[R,C];
  U:Real[R,R];
  α:Real <- simulate_uniform(2.0, 10.0);
  v:Real[C];

  for i in 1..R {
    for j in 1..C {
      M[i,j] <- simulate_uniform(-10.0, 10.0);
    }
  }
  for i in 1..R {
    for j in 1..R {
      U[i,j] <- simulate_uniform(-2.0, 2.0);
    }
  }
  for i in 1..C {
    v[i] <- pow(simulate_uniform(-2.0, 2.0), 2.0);
  }
  U <- U*transpose(U);

  σ2:IndependentInverseGamma(box(α), box(v));
  π:MatrixNormalInverseGamma(box(M), box(llt(U)), σ2);
  test_pdf(π, R, C, N, B, S);
}
