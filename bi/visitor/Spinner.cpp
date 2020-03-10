/**
 * @file
 */
#include "bi/visitor/Spinner.hpp"

#include "bi/visitor/all.hpp"

bi::Statement* bi::Spinner::modify(ExpressionStatement* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->single);
  if (loops) {
    return new Braces(new StatementList(loops, o, o->loc), o->loc);
  } else {
    return o;
  }
}

bi::Statement* bi::Spinner::modify(Assume* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->left);
  loops = extract(o->right, loops);
  if (loops) {
    return new Braces(new StatementList(loops, o, o->loc), o->loc);
  } else {
    return o;
  }
}

bi::Statement* bi::Spinner::modify(LocalVariable* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->brackets);
  loops = extract(o->args, loops);
  loops = extract(o->value, loops);
  if (loops) {
    return new StatementList(loops, o, o->loc);
  } else {
    return o;
  }
}

bi::Statement* bi::Spinner::modify(If* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->cond);
  if (loops) {
    return new Braces(new StatementList(loops, o, o->loc), o->loc);
  } else {
    return o;
  }
}

bi::Statement* bi::Spinner::modify(For* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->from);
  loops = extract(o->to, loops);
  if (loops) {
    return new Braces(new StatementList(loops, o, o->loc), o->loc);
  } else {
    return o;
  }
}

bi::Statement* bi::Spinner::modify(Parallel* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->from);
  loops = extract(o->to, loops);
  if (loops) {
    return new Braces(new StatementList(loops, o, o->loc), o->loc);
  } else {
    return o;
  }
}

bi::Statement* bi::Spinner::modify(While* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->cond);
  if (loops) {
    return new Braces(new StatementList(loops, o, o->loc), o->loc);
  } else {
    return o;
  }
}

bi::Statement* bi::Spinner::modify(DoWhile* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->cond);
  if (loops) {
    o->braces = new Braces(new StatementList(o->braces->strip(), loops,
        o->loc), o->loc);
  }
  return o;
}

bi::Statement* bi::Spinner::modify(Assert* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->cond);
  if (loops) {
    return new Braces(new StatementList(loops, o, o->loc), o->loc);
  } else {
    return o;
  }
}

bi::Statement* bi::Spinner::modify(Return* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->single);
  if (loops) {
    return new Braces(new StatementList(loops, o, o->loc), o->loc);
  } else {
    return o;
  }
}

bi::Statement* bi::Spinner::modify(Yield* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->single);
  if (loops) {
    return new StatementList(loops, o, o->loc);
  } else {
    return o;
  }
}

bi::Statement* bi::Spinner::extract(Expression* o, Statement* loops) {
  /* gather all spins */
  Gatherer<Spin> spins;
  o->accept(&spins);

  /* construct loop for each spin */
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
  return loops;
}
