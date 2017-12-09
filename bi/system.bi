/**
 * Execute a command.
 *
 *   - cmd: The command string.
 *
 * Returns the return value of the execution.
 */
function system(cmd:String) -> Integer32 {
  cpp{{
  return std::system(cmd_.c_str());
  }}
}
