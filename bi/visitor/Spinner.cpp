/**
 * @file
 */
#include "bi/visitor/Spinner.hpp"

#include "bi/visitor/all.hpp"

bi::Statement* bi::Spinner::modify(ExpressionStatement* o) {
  Gatherer<Spin> spins;
  o->single->accept(&spins);

  Statement* loops = nullptr;
  for (auto spin : spins) {
    auto call = dynamic_cast<Call*>(spin->single);
    if (call) {
      /* temporary variable to hold fiber handle */
      auto name = new Name();
      auto var = new LocalVariable(AUTO, name, new EmptyType(spin->loc),
          new EmptyExpression(spin->loc), new EmptyExpression(spin->loc),
          spin->single, spin->loc);

      /* loop to run fiber to completion and yield values along the way */
      auto query = new Query(new NamedExpression(name, spin->loc), spin->loc);
      auto get = new Get(new NamedExpression(name, spin->loc), spin->loc);
      auto yield = new Braces(new Yield(get, spin->loc), spin->loc);
      auto loop = new While(query, yield, spin->loc);
      auto block = new StatementList(var, loop, spin->loc);

      /* accumulate loops */
      if (!loops) {
        loops = block;
      } else {
        loops = new StatementList(loops, block, spin->loc);
      }

      /* replace the spin on the call with a spin on the temporary variable */
      spin->single = new NamedExpression(name, spin->loc);
    }
  }
  if (loops) {
    return new StatementList(loops, o, o->loc);
  } else {
    return o;
  }
}
