/*
 * Returns the file associated with stdin.
 */
function getStdIn() -> File {
  cpp{{
  return stdin;
  }}
}

/*
 * Returns the file associated with stdout.
 */
function getStdOut() -> File {
  cpp{{
  return stdout;
  }}
}
/*
 * Returns the file associated with stderr.
 */
function getStdErr() -> File {
  cpp{{
  return stderr;
  }}
}

/**
 * Input stream for stdin.
 */
stdin:InputStream' <- InputStream(getStdIn());

/**
 * Output stream for stdout.
 */
stdout:OutputStream' <- OutputStream(getStdOut());

/**
 * Output stream for stderr.
 */
stderr:OutputStream' <- OutputStream(getStdErr());
