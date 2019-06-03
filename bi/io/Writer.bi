/**
 * Abstract writer.
 *
 * Typical use is to use the `Writer` factory function to instantiate an
 * object of an appropriate derived class based on the file extension of the
 * given path:
 *
 *     auto writer <- Writer(path);
 *
 * A write of a single buffer can then be performed with:
 *
 *     writer.write(buffer);
 *
 * A write of a sequence of buffers can be performed with:
 *
 *     writer.startSequence();
 *     writer.write(buffer1);
 *     writer.write(buffer2);
 *     writer.write(buffer3);
 *     writer.endSequence();
 *
 * which is useful for not keeping the entire contents of the file, to be
 * written, in memory.
 *
 * Finally, close the file:
 *
 *     writer.close();
 *
 * A file may not be valid until the writer is closed, depending on the file
 * format.
 */
class Writer {
  /**
   * Open a file.
   *
   * - path : Path of the file.
   */
  function open(path:String);
  
  /**
   * Write the entire contents of the file.
   *
   * - buffer: Buffer to write.
   */
  function write(buffer:MemoryBuffer);
  
  /**
   * Flush accumulated writes to the file.
   */
  function flush();
  
  /**
   * Close the file.
   */
  function close();

  /**
   * Start a mapping.
   */
  function startMapping();
  
  /**
   * End a mapping.
   */
  function endMapping();
  
  /**
   * Start a sequence.
   */
  function startSequence();
  
  /**
   * End a sequence.
   */
  function endSequence();

  function visit(value:ObjectValue);
  function visit(value:ArrayValue);
  function visit(value:NilValue);
  function visit(value:BooleanValue);
  function visit(value:IntegerValue);
  function visit(value:RealValue);
  function visit(value:StringValue);
  function visit(value:BooleanVectorValue);
  function visit(value:IntegerVectorValue);
  function visit(value:RealVectorValue);
  function visit(value:BooleanMatrixValue);
  function visit(value:IntegerMatrixValue);
  function visit(value:RealMatrixValue);
}

/**
 * Create a writer for a file.
 *
 * - path: Path of the file.
 *
 * Returns: the writer.
 *
 * The file extension of `path` is used to determine the precise type of the
 * returned object. Supported file extension are `.json` and `.yml`.
 */
function Writer(path:String) -> Writer {
  auto ext <- extension(path);
  if ext == ".json" {
    writer:JSONWriter;
    writer.open(path);
    return writer;
  } else if ext == ".yml" {
    writer:YAMLWriter;
    writer.open(path);
    return writer;
  } else {
    error("unrecognized file extension '" + ext + "' in path '" + path +
        "'; supported extensions are '.json' and '.yml'.");
  }
}
