/**
 * Output an error message and exit.
 */
function error(msg:String) {
  stderr.print("error: " + msg + "\n");
  exit(1);
}

/**
 * Output a warning message.
 */
function warning(msg:String) {
  stderr.print("warning: " + msg + "\n");
}
