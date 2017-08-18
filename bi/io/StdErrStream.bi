import basic;
import io.OutputStream;

/**
 * Output stream for stderr.
 */
class StdErrStream < OutputStream(getStdErr()) {
  //
}

/*
 * Returns the file associated with stderr.
 */
function getStdErr() -> File {
  cpp{{
  return ::stderr;
  }}
}

/**
 * Standard error.
 */
stderr:StdErrStream;
