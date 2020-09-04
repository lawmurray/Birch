class Global {
  /**
   * Lower corner of domain of interest.
   */
  l:Real[_];

  /**
   * Upper corner of domain of interest.
   */
  u:Real[_];

  /**
   * Probability of detection.
   */
  d:Real;

  /**
   * Initial value covariance.
   */
  M:Real[_,_];
  
  /**
   * Transition matrix.
   */
  A:Real[_,_];

  /**
   * State noise covariance matrix.
   */
  Q:Real[_,_];

  /**
   * Observation matrix.
   */
  B:Real[_,_];

  /**
   * Observation noise covariance matrix.
   */
  R:Real[_,_];

  /**
   * Birth rate.
   */
  λ:Real;

  /**
   * Clutter rate.
   */
  μ:Real;

  /**
   * Track length rate.
   */
  τ:Real;
  
  function read(buffer:Buffer) {
    l <- buffer.get("l", l)!;
    u <- buffer.get("u", u)!;
    d <- buffer.get("d", d)!;
    M <- buffer.get("M", M)!;
    A <- buffer.get("A", A)!;
    Q <- buffer.get("Q", Q)!;
    B <- buffer.get("B", B)!;
    R <- buffer.get("R", R)!;
    λ <- buffer.get("λ", λ)!;
    μ <- buffer.get("μ", μ)!;
    τ <- buffer.get("τ", τ)!;
  }
  
  function write(buffer:Buffer) {
    buffer.set("l", l);
    buffer.set("u", u);
    buffer.set("d", d);
    buffer.set("M", M);
    buffer.set("A", A);
    buffer.set("Q", Q);
    buffer.set("B", B);
    buffer.set("R", R);
    buffer.set("λ", λ);
    buffer.set("μ", μ);
    buffer.set("τ", τ);
  }
}