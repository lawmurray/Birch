/**
 * Read numbers from a file.
 */
function read(file:String, N:Integer) -> Real[_] {
  cpp{{
  std::ifstream stream(file_);
  }}
  x:Real[N];
  v:Real;
  for (n:Integer in 1..N) {
    cpp{{
    stream >> v_;
    }}
    x[n] <- v;
  }
  return x;
}
