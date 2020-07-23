/**
 * Execute a command.
 *
 *   - cmd: The command string.
 *
 * Return: the return code of the execution.
 */
function system(cmd:String) -> Integer {
  cpp{{
  int code;
  int status = std::system(cmd.c_str());
  if (WIFEXITED(status)) {
    code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    code = WTERMSIG(status);
  } else if (WIFSTOPPED(status)) {
    code = WSTOPSIG(status);
  } else {
    code = status;
  }
  #ifdef HAVE_LIBEXPLAIN_SYSTEM_H
  if (code != 0) {
    ::fprintf(::stderr, "%s\n", ::explain_system(cmd.c_str()));
  }
  #endif
  return code;
  }}
}

/**
 * Exit.
 *
 *   - code: An exit code.
 */
function exit(code:Integer) {
  cpp{{
  std::exit(code);
  }}
}
