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
    success:Boolean;
    cpp{{
    std::ofstream stream(path_);
    success_ = stream.is_open();
    }}
    if (success) {
      cpp{{
      libubjpp::JSONGenerator generator(stream);
      generator.write(top);
      }}
    } else {
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
