/**
 * Execute a command.
 *
 *   - cmd: The command string.
 *
 * Return: the return code of the execution.
 */
function system(cmd:String) -> Integer {
  cpp{{
  int ret = std::system(cmd_.c_str());
  return WEXITSTATUS(ret);
  }}
}

/**
 * Exit.
 *
 *   - code: An exit code.
 */
function exit(code:Integer) {
  cpp{{
  std::exit(code_);
  }}
}
