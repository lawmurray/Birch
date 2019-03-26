/**
 * Abstract generator.
 */
class Generator {
  /**
   * Generate a file.
   *
   * - path: File path.
   * - buffer: Buffer from which to load contents.
   */
  function generate(path:String, buffer:MemoryBuffer);

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
