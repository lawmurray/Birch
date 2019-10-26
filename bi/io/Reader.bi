/**
 * Abstract reader.
 *
 * Typical use is to use the `Reader` factory function to instantiate an
 * object of an appropriate derived class based on the file extension of the
 * given path:
 *
 *     auto reader <- Reader(path);
 *
 * A reader of a single buffer can then be performed with:
 *
 *     reader.read(buffer);
 *
 * Finally, close the file:
 *
 *     reader.close();
 */
class Reader {  
  /**
   * Open a file.
   *
   * - path : Path of the file.
   */
  function open(path:String);
  
  /**
   * Read the entire contents of the file.
   *
   * - buffer: Buffer into which to read.
   */
  function read(buffer:MemoryBuffer);
  
  /**
   * Close the file.
   */
  function close();
}

/**
 * Create a reader for a file.
 *
 * - path: Path of the file.
 *
 * Returns: the reader.
 *
 * The file extension of `path` is used to determine the precise type of the
 * returned object. Supported file extension are `.json` and `.yml`.
 */
function Reader(path:String) -> Reader {
  auto ext <- extension(path);
  result:Reader?;
  if ext == ".json" {
    reader:JSONReader;
    reader.open(path);
    result <- reader;
  } else if ext == ".yml" {
    reader:YAMLReader;
    reader.open(path);
    result <- reader;
  }
  if !result? {
    error("unrecognized file extension '" + ext + "' in path '" + path +
        "'; supported extensions are '.json' and '.yml'.");
  }
  return result!;
}
