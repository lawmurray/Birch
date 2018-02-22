/**
 * JSON file writer.
 */
class JSONWriter < MemoryWriter {
  hpp{{
  libubjpp::value top;
  }}

  /**
   * File path.
   */
  path:String;
  
  /**
   * Constructor.
   */
  function make(path:String) {
    this.path <- path;
    cpp{{
    group = &top;
    }}
  }
  
  /**
   * Save data to file.
   */
  function save() {
    success:Boolean <- false;
    cpp{{
    std::ofstream stream(path_);
    if (stream.is_open()) {
      libubjpp::JSONGenerator generator(stream);
      generator.write(top);
      success_ = true;
    }
    }}
    if (!success) {
      stderr.print("warning: could not save to \'" + path + "'\n");
    }
  }
}

/**
 * Factory function.
 */
function JSONWriter(path:String) -> JSONWriter {
  writer:JSONWriter;
  writer.make(path);
  return writer;
}
