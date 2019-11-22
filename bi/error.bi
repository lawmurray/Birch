/**
 * Output an error message and exit.
 */
function error(msg:String) {
  cpp{{
  libbirch::abort(msg, 1);
  }}
}

/**
 * Output a warning message.
 */
function warn(msg:String) {
  stderr.print("warning: " + msg + "\n");
}
