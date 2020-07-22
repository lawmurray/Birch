/*
 * Test matrix normal-inverse-Wishart pdf evaluations.
 */
program test_pdf_matrix_normal_inverse_wishart(R:Integer <- 3, C:Integer <- 2,
    N:Integer <- 10000, B:Integer <- 500, S:Integer <- 30) {
  M:Real[R,C];
  U:Real[R,R];
  k:Real;
  Ψ:Real[C,C];

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
  k <- simulate_uniform(C - 1.0, C + 9.0);
  for i in 1..C {
    for j in 1..C {
      Ψ[i,j] <- simulate_uniform(-2.0, 2.0);
    }
  }
  U <- U*transpose(U);
  Ψ <- Ψ*transpose(Ψ);

  V:InverseWishart(box(llt(Ψ)), box(k));
  π:MatrixNormalInverseWishart(box(M), box(llt(U)), V);
  test_pdf(π, R, C, N, B, S);
}
