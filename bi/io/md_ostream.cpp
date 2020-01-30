/**
 * @file
 */
#include "bi/io/md_ostream.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/encode.hpp"

bi::md_ostream::md_ostream(std::ostream& base) :
    bih_ostream(base),
    package(nullptr),
    depth(1) {
  //
}

void bi::md_ostream::visit(const Package* o) {
  package = o;

  /* collect type names, in order to distinguish between factory functions
   * and other functions */
  std::unordered_set<std::string> classNames, basicNames;
  Gatherer<Class> allClasses;
  o->accept(&allClasses);
  Gatherer<Basic> allBasics;
  o->accept(&allBasics);
  for (auto o : allClasses) {
    classNames.insert(o->name->str());
  }
  for (auto o : allBasics) {
    basicNames.insert(o->name->str());
  }

  /* lambdas */
  auto all = [](const void*) {
    return true;
  };
  auto docsNotEmpty = [](const Located* o) {
    return !o->loc->doc.empty();
  };
  auto docsNotEmptyAndNotFactory = [&](const Function* o) {
    return !o->loc->doc.empty() &&
        basicNames.find(o->name->str()) == basicNames.end() &&
        classNames.find(o->name->str()) == classNames.end();
  };
  auto docsNotEmptyAndBasicFactory = [&](const Function* o) {
    return !o->loc->doc.empty() &&
        basicNames.find(o->name->str()) != basicNames.end();
  };
  auto sortByName = [](const Named* o1, const Named* o2) {
    return o1->name->str() < o2->name->str();
  };

  /* basic types */
  Gatherer<Basic> basics(docsNotEmpty, false);
  o->accept(&basics);
  std::stable_sort(basics.begin(), basics.end(), sortByName);
  if (basics.size() > 0) {
    genHead("Types");
    line("| Name | Description |");
    line("| --- | --- |");
    ++depth;
    for (auto o : basics) {
      line("| " << o << " | " << one_line(o->loc->doc) << " |");
    }
    line("");

    Gatherer<Function> basicFactories(docsNotEmptyAndBasicFactory, false);
    o->accept(&basicFactories);
    std::stable_sort(basicFactories.begin(), basicFactories.end(), sortByName);

    if (basicFactories.size() > 0) {
      ++depth;
      for (auto o : basicFactories) {
        *this << o;
        line("");
        line(quote(detailed(o->loc->doc), "    "));
        line("");
      }
      --depth;
    }
    --depth;
  }

  /* global variables */
  Gatherer<GlobalVariable> variables(docsNotEmpty, false);
  o->accept(&variables);
  std::stable_sort(variables.begin(), variables.end(), sortByName);
  if (variables.size() > 0) {
    genHead("Variables");
    line("| Name | Description |");
    line("| --- | --- |");
    ++depth;
    for (auto o : variables) {
      line("| " << o << " | " << one_line(o->loc->doc) << " |");
    }
    line("");
    --depth;
  }

  /* programs */
  Gatherer<Program> programs(docsNotEmpty, false);
  o->accept(&programs);
  std::stable_sort(programs.begin(), programs.end(), sortByName);
  if (programs.size() > 0) {
    genHead("Programs");
    ++depth;
    std::string desc;
    for (auto o : programs) {
      genHead(o->name->str());
      line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");
      desc = quote(detailed(o->loc->doc), "    ");
      *this << o;
      line("");
      line(desc);
      line("");
    }
    line("");
    --depth;
  }

  /* functions */
  Gatherer<Function> functions(docsNotEmptyAndNotFactory, false);
  o->accept(&functions);
  std::stable_sort(functions.begin(), functions.end(), sortByName);
  if (functions.size() > 0) {
    genHead("Functions");
    ++depth;
    std::string name, desc;
    for (auto o : functions) {
      if (o->name->str() != name) {
        /* heading only for the first overload of this name */
        genHead(o->name->str());
        line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");
      }
      name = o->name->str();
      desc = quote(detailed(o->loc->doc), "    ");
      *this << o;
      line("");
      line(desc);
      line("");
    }
    line("");
    --depth;
  }

  /* fibers */
  Gatherer<Fiber> fibers(docsNotEmpty, false);
  o->accept(&fibers);
  std::stable_sort(fibers.begin(), fibers.end(), sortByName);
  if (fibers.size() > 0) {
    genHead("Fibers");
    ++depth;
    std::string name, desc;
    for (auto o : fibers) {
      if (o->name->str() != name) {
        /* heading only for the first overload of this name */
        genHead(o->name->str());
        line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");
      }
      name = o->name->str();
      desc = quote(detailed(o->loc->doc), "    ");
      *this << o;
      line("");
      line(desc);
      line("");
    }
    line("");
    --depth;
  }

  /* unary operators */
  Gatherer<UnaryOperator> unaries(all, false);
  o->accept(&unaries);
  std::stable_sort(unaries.begin(), unaries.end(), sortByName);
  if (unaries.size() > 0) {
    genHead("Unary Operators");
    ++depth;
    std::string name, desc;
    for (auto o : unaries) {
      if (o->name->str() != name) {
        /* heading only for the first overload of this name */
        genHead(o->name->str());
        line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");
      }
      name = o->name->str();
      desc = quote(detailed(o->loc->doc), "    ");
      *this << o;
      line("");
      line(desc);
      line("");
    }
    line("");
    --depth;
  }

  /* binary operators */
  Gatherer<BinaryOperator> binaries(all, false);
  o->accept(&binaries);
  std::stable_sort(binaries.begin(), binaries.end(), sortByName);
  if (binaries.size() > 0) {
    genHead("Binary Operators");
    ++depth;
    std::string name, desc;
    for (auto o : binaries) {
      if (o->name->str() != name) {
        /* heading only for the first overload of this name */
        genHead(o->name->str());
        line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");
      }
      name = o->name->str();
      desc = quote(detailed(o->loc->doc), "    ");
      *this << o;
      line("");
      line(desc);
      line("");
    }
    line("");
    --depth;
  }

  /* classes */
  Gatherer<Class> classes(docsNotEmpty, false);
  o->accept(&classes);
  std::stable_sort(classes.begin(), classes.end(), sortByName);
  if (classes.size() > 0) {
    genHead("Classes");
    ++depth;
    for (auto o : classes) {
      *this << o;
    }
    --depth;
  }
}

void bi::md_ostream::visit(const Name* o) {
  middle(o->str());
}

void bi::md_ostream::visit(const Parameter* o) {
  middle(o->name << ':' << o->type);
  if (!o->value->isEmpty()) {
    *this << " <- " << o->value;
  }
}

void bi::md_ostream::visit(const GlobalVariable* o) {
  middle(o->name << ':' << o->type);
}

void bi::md_ostream::visit(const MemberVariable* o) {
  middle(o->name << ':' << o->type);
}

void bi::md_ostream::visit(const Function* o) {
  start("!!! quote \"function " << o->name);
  if (o->isGeneric()) {
    middle("&lt;" << o->typeParams << "&gt;");
  }
  middle('(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void bi::md_ostream::visit(const Fiber* o) {
  start("!!! quote \"fiber " << o->name);
  if (o->isGeneric()) {
    middle("&lt;" << o->typeParams << "&gt;");
  }
  middle('(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void bi::md_ostream::visit(const Program* o) {
  start("!!! quote \"program " << o->name << '(' << o->params << ")\"");
}

void bi::md_ostream::visit(const MemberFunction* o) {
  start("!!! quote \"");
  if (o->has(ABSTRACT)) {
    middle("abstract ");
  }
  if (o->has(FINAL)) {
    middle("final ");
  }
  middle("function");
  middle(' ' << o->name << '(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void bi::md_ostream::visit(const MemberFiber* o) {
  start("!!! quote \"");
  if (o->has(ABSTRACT)) {
    middle("abstract ");
  }
  if (o->has(FINAL)) {
    middle("final ");
  }
  middle("fiber");
  middle(' ' << o->name << '(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void bi::md_ostream::visit(const BinaryOperator* o) {
  start("!!! quote \"operator (");
  middle(o->left << ' ' << o->name << ' ' << o->right << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void bi::md_ostream::visit(const UnaryOperator* o) {
  start("!!! quote \"operator (");
  middle(o->name << ' ' << o->single << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void bi::md_ostream::visit(const AssignmentOperator* o) {
  middle(o->single->type);
}

void bi::md_ostream::visit(const ConversionOperator* o) {
  middle(o->returnType);
}

void bi::md_ostream::visit(const Basic* o) {
  middle(o->name);
  if (!o->base->isEmpty()) {
    if (o->isAlias()) {
      middle(" = ");
    } else {
      middle(" < ");
    }
    middle(o->base);
  }
}

void bi::md_ostream::visit(const Class* o) {
  /* lambdas */
  auto docsNotEmpty = [](const Located* o) {
    return !o->loc->doc.empty();
  };
  auto isFactory = [o](const Function* func) {
    return func->name->str() == o->name->str();
  };
  auto sortByName = [](const Named* o1, const Named* o2) {
    return o1->name->str() < o2->name->str();
  };

  /* anchor for internal links */
  genHead(o->name->str());
  line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");
  start("!!! quote \"");
  if (o->has(ABSTRACT)) {
    middle("abstract ");
  }
  if (o->has(FINAL)) {
    middle("final ");
  }
  middle("class " << o->name);
  if (o->isGeneric()) {
    middle("&lt;" << o->typeParams << "&gt;");
  }
  if (!o->isAlias() && !o->params->isEmpty()) {
    middle('(' << o->params << ')');
  }
  if (!o->base->isEmpty()) {
    if (o->isAlias()) {
      middle(" = ");
    } else {
      middle(" < ");
    }
    middle(o->base);
    if (!o->args->isEmpty()) {
      middle('(' << o->args << ')');
    }
  }
  finish("\"\n");
  line(quote(detailed(o->loc->doc), "    ") << "\n");

  ++depth;

  /* factory functions */
  Gatherer<Function> factories(isFactory, false);
  if (package) {
    package->accept(&factories);
    if (factories.size() > 0) {
      genHead("Factory Functions");
      line("| Name | Description |");
      line("| --- | --- |");
      ++depth;
      for (auto o : factories) {
        start("| ");
        middle('[' << o->name->str() << ']');
        middle("(#" << anchor(o->name->str()) << ')');
        finish(" | " << brief(o->loc->doc) << " |");
      }
      line("");
      --depth;
    }
  }

  /* assignment operators */
  Gatherer<AssignmentOperator> assignments;
  o->accept(&assignments);
  if (assignments.size() > 0) {
    genHead("Assignments");
    line("| Name | Description |");
    line("| --- | --- |");
    ++depth;
    for (auto o : assignments) {
      line("| " << o << " | " << one_line(o->loc->doc) << " |");
    }
    line("");
    --depth;
  }

  /* conversion operators */
  Gatherer<ConversionOperator> conversions;
  o->accept(&conversions);
  if (conversions.size() > 0) {
    genHead("Conversions");
    line("| Name | Description |");
    line("| --- | --- |");
    ++depth;
    for (auto o : conversions) {
      line("| " << o << " | " << one_line(o->loc->doc) << " |");
    }
    line("");
    --depth;
  }

  /* member variables */
  Gatherer<MemberVariable> variables(docsNotEmpty);
  o->accept(&variables);
  if (variables.size() > 0) {
    genHead("Member Variables");
    line("| Name | Description |");
    line("| --- | --- |");
    ++depth;
    for (auto o : variables) {
      line("| " << o << " | " << one_line(o->loc->doc) << " |");
    }
    line("");
    --depth;
  }

  /* member functions */
  Gatherer<MemberFunction> functions(docsNotEmpty);
  o->accept(&functions);
  if (functions.size() > 0) {
    genHead("Member Functions");
    line("| Name | Description |");
    line("| --- | --- |");
    ++depth;
    for (auto o : functions) {
      start("| ");
      middle('[' << o->name->str() << ']');
      middle("(#" << anchor(o->name->str()) << ')');
      finish(" | " << brief(o->loc->doc) << " |");
    }
    line("");
    --depth;
  }

  /* member fibers */
  Gatherer<MemberFiber> fibers(docsNotEmpty);
  o->accept(&fibers);
  if (fibers.size() > 0) {
    genHead("Member Fibers");
    line("| Name | Description |");
    line("| --- | --- |");
    ++depth;
    for (auto o : fibers) {
      start("| ");
      middle('[' << o->name->str() << ']');
      middle("(#" << anchor(o->name->str()) << ')');
      finish(" | " << brief(o->loc->doc) << " |");
    }
    line("");
    --depth;
  }

  /* factory function details */
  if (factories.size() > 0) {
    genHead("Factory Function Details");
    ++depth;
    for (auto o : factories) {
      *this << o;
      line("");
      line(quote(detailed(o->loc->doc), "    "));
      line("");
    }
    line("");
    --depth;
  }

  /* member function details */
  std::stable_sort(functions.begin(), functions.end(), sortByName);
  if (functions.size() > 0) {
    genHead("Member Function Details");
    ++depth;
    std::string name, desc;
    for (auto o : functions) {
      if (o->name->str() != name) {
        /* heading only for the first overload of this name */
        genHead(o->name->str());
        line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");
      }
      name = o->name->str();
      desc = quote(detailed(o->loc->doc), "    ");
      *this << o;
      line("");
      line(desc);
      line("");
    }
    line("");
    --depth;
  }

  /* member fiber details */
  std::stable_sort(fibers.begin(), fibers.end(), sortByName);
  if (fibers.size() > 0) {
    genHead("Member Fiber Details");
    ++depth;
    std::string name, desc;
    for (auto o : fibers) {
      if (o->name->str() != name) {
        /* heading only for the first overload of this name */
        genHead(o->name->str());
        line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");
      }
      name = o->name->str();
      desc = quote(detailed(o->loc->doc), "    ");
      *this << o;
      line("");
      line(desc);
      line("");
    }
    line("");
    --depth;
  }
  --depth;
}

void bi::md_ostream::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}

void bi::md_ostream::visit(const NamedType* o) {
  ///@todo
  //if (o->category == CLASS) {
    middle('[' << o->name << "](../classes/" << o->name->str() << ".md)");
  //} else {
  //  middle('[' << o->name << "](../types/index.md#" << o->name->str() << ')');
  //}
  if (!o->typeArgs->isEmpty()) {
    middle("&lt;" << o->typeArgs << "&gt;");
  }
  if (o->weak) {
    middle("&amp;");
  }
}

void bi::md_ostream::visit(const ArrayType* o) {
  middle(o->single << '[');
  for (int i = 0; i < o->depth(); ++i) {
    if (i > 0) {
      middle(',');
    }
    middle("\\_");
  }
  middle(']');
}

void bi::md_ostream::visit(const TupleType* o) {
  middle('(' << o->single << ')');
}

void bi::md_ostream::visit(const FunctionType* o) {
  middle('@' << '(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
}

void bi::md_ostream::visit(const FiberType* o) {
  middle(o->yieldType << '!' << o->returnType);
}

void bi::md_ostream::visit(const OptionalType* o) {
  middle(o->single << '?');
}

void bi::md_ostream::genHead(const std::string& name) {
  finish("");
  for (int i = 0; i < depth; ++i) {
    middle('#');
  }
  middle(' ');
  finish(name);
  line("");
}
