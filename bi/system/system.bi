/**
 * Execute a command.
 *
 *   - cmd: The command string.
 *
 * Return: the return code of the execution.
 */
function system(cmd:String) -> Integer {
  cpp{{
  int status = std::system(cmd_.c_str());
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    return WTERMSIG(status);
  } else if (WIFSTOPPED(status)) {
    return WSTOPSIG(status);
  }
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
