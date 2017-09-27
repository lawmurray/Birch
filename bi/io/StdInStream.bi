/**
 * Input stream for stdin.
 */
class StdInStream < InputStream(getStdIn()) {
  //
}

/*
 * Returns the file associated with stdin.
 */
function getStdIn() -> File {
  cpp{{
  return ::stdin;
  }}
}

/**
 * Standard input.
 */
stdin:StdInStream;
