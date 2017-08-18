import basic;
import io.OutputStream;

/**
 * Output stream for stdout.
 */
class StdOutStream < OutputStream(getStdOut()) {
  //
}

/*
 * Returns the file associated with stdout.
 */
function getStdOut() -> File {
  cpp{{
  return ::stdout;
  }}
}

/**
 * Standard output.
 */
stdout:StdOutStream;
