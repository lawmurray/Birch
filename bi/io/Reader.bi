/**
 * Abstract reader.
 *
 * Typical use is to use the `Reader` factory function to instantiate an
 * object of an appropriate derived class based on the file extension of the
 * given path:
 *
 *     auto reader <- Reader(path);
 *
 * The whole contents of the file can then be read into a buffer with:
 *
 *     auto buffer <- reader.scan();
 *
 * Finally, close the file:
 *
 *     reader.close();
 */
abstract class Reader {  
  /**
   * Open a file.
   *
   * - path: Path of the file.
   */
  abstract function open(path:String);
  
  /**
   * Read the entire contents of the file.
   *
   * - buffer: Buffer into which to read.
   */
  abstract function scan() -> MemoryBuffer;

  /**
   * Read the contents of the file sequentially.
   *
   * Yields: A buffer for each element of the top level sequence.
   * 
   * For a file that consists of a sequence at the top level, yields each
   * element of that sequence one at a time, to avoid reading the whole file
   * into memory at once.
   */
  abstract fiber walk() -> Buffer;
  
  /**
   * Close the file.
   */
  abstract function close();
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
