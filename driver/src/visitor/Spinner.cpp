/**
 * @file
 */
#include "src/visitor/Spinner.hpp"

#include "src/visitor/all.hpp"
#include "src/exception/all.hpp"

birch::Statement* birch::Spinner::modify(ExpressionStatement* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->single);
  if (loops) {
    return new StatementList(loops, o, o->loc);
  } else {
    return o;
  }
}

birch::Statement* birch::Spinner::modify(Assume* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->left);
  loops = extract(o->right, loops);
  if (loops) {
    return new StatementList(loops, o, o->loc);
  } else {
    return o;
  }
}

birch::Statement* birch::Spinner::modify(LocalVariable* o) {
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

birch::Statement* birch::Spinner::modify(If* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->cond);
  if (loops) {
    return new StatementList(loops, o, o->loc);
  } else {
    return o;
  }
}

birch::Statement* birch::Spinner::modify(For* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->from);
  loops = extract(o->to, loops);
  if (loops) {
    return new StatementList(loops, o, o->loc);
  } else {
    return o;
  }
}

birch::Statement* birch::Spinner::modify(Parallel* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->from);
  loops = extract(o->to, loops);
  if (loops) {
    return new StatementList(loops, o, o->loc);
  } else {
    return o;
  }
}

birch::Statement* birch::Spinner::modify(While* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->cond);
  if (loops) {
    return new StatementList(loops, o, o->loc);
  } else {
    return o;
  }
}

birch::Statement* birch::Spinner::modify(DoWhile* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->cond);
  if (loops) {
    /* for a do-while loop, the extra statements need to be inserted at the
     * end of the loop body, just before the condition */
    o->braces = new Braces(new StatementList(o->braces->strip(), loops,
        o->loc), o->loc);
  }
  return o;
}

birch::Statement* birch::Spinner::modify(Assert* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->cond);
  if (loops) {
    return new StatementList(loops, o, o->loc);
  } else {
    return o;
  }
}

birch::Statement* birch::Spinner::modify(Return* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->single);
  if (loops) {
    return new StatementList(loops, o, o->loc);
  } else {
    return o;
  }
}

birch::Statement* birch::Spinner::modify(Yield* o) {
  ContextualModifier::modify(o);
  auto loops = extract(o->single);
  if (loops) {
    return new StatementList(loops, o, o->loc);
  } else {
    return o;
  }
}

birch::Statement* birch::Spinner::extract(Expression* o, Statement* loops) {
  /* gather all spins */
  Gatherer<Spin> spins;
  o->accept(&spins);

  /* construct loop for each spin */
  for (auto spin : spins) {
    if (!currentFiber) {
      throw SpinException(spin);
    }
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
      auto yield = new Yield(get, spin->loc);
      auto loop = new While(query, new Braces(yield, spin->loc), spin->loc);
      auto pre = new StatementList(var, loop, spin->loc);

      /* accumulate loops */
      if (!loops) {
        loops = pre;
      } else {
        loops = new StatementList(loops, pre, spin->loc);
      }

      /* replace the spin on the call with a spin on the temporary variable */
      spin->single = new NamedExpression(name, spin->loc);
    }
  }
  return loops;
}
