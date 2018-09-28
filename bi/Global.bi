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
  
  function read(reader:Reader) {
    l <- reader.get("l", l)!;
    u <- reader.get("u", u)!;
    d <- reader.get("d", d)!;
    M <- reader.get("M", M)!;
    A <- reader.get("A", A)!;
    Q <- reader.get("Q", Q)!;
    B <- reader.get("B", B)!;
    R <- reader.get("R", R)!;
    λ <- reader.get("λ", λ)!;
    μ <- reader.get("μ", μ)!;
    τ <- reader.get("τ", τ)!;
  }
  
  function write(writer:Writer) {
    writer.set("l", l);
    writer.set("u", u);
    writer.set("d", d);
    writer.set("M", M);
    writer.set("A", A);
    writer.set("Q", Q);
    writer.set("B", B);
    writer.set("R", R);
    writer.set("λ", λ);
    writer.set("μ", μ);
    writer.set("τ", τ);
  }
}