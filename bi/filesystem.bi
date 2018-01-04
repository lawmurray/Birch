/**
 * Create a directory.
 */
function mkdir(path:String) {
  cpp{{
  boost::filesystem::create_directories(path_);
  }}
}
