/**
 * @file
 */
#include "src/generate/MarkdownGenerator.hpp"

#include "src/visitor/Gatherer.hpp"
#include "src/primitive/string.hpp"

birch::MarkdownGenerator::MarkdownGenerator(std::ostream& base) :
    BirchGenerator(base, 0, true),
    package(nullptr),
    depth(1) {
  //
}

void birch::MarkdownGenerator::visit(const Package* o) {
  package = o;

  /* lambdas */
  auto all = [](const void*) {
    return true;
  };
  auto docsNotEmpty = [](const Located* o) {
    return !o->loc->doc.empty();
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
    --depth;
    line("");
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
      desc = detailed(o->loc->doc);
      *this << o;
      line("");
      line(desc);
      line("");
    }
    line("");
    --depth;
  }

  /* functions */
  Gatherer<Function> functions(docsNotEmpty, false);
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

  /* operators */
  Gatherer<BinaryOperator> binaries(all, false);
  o->accept(&binaries);
  std::stable_sort(binaries.begin(), binaries.end(), sortByName);

  Gatherer<UnaryOperator> unaries(all, false);
  o->accept(&unaries);
  std::stable_sort(unaries.begin(), unaries.end(), sortByName);

  if (binaries.size() + unaries.size() > 0) {
    genHead("Operators");
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

  /* structs */
  Gatherer<Struct> structs(docsNotEmpty, false);
  o->accept(&structs);
  std::stable_sort(structs.begin(), structs.end(), sortByName);
  genHead("Structs");
  ++depth;
  for (auto o : structs) {
    *this << o;
  }
  --depth;

  /* classes */
  Gatherer<Class> classes(docsNotEmpty, false);
  o->accept(&classes);
  std::stable_sort(classes.begin(), classes.end(), sortByName);
  genHead("Classes");
  ++depth;
  for (auto o : classes) {
    *this << o;
  }
  --depth;
}

void birch::MarkdownGenerator::visit(const Name* o) {
  middle(o->str());
}

void birch::MarkdownGenerator::visit(const Parameter* o) {
  middle(o->name << ':' << o->type);
  if (!o->value->isEmpty()) {
    *this << " <- " << o->value;
  }
}

void birch::MarkdownGenerator::visit(const GlobalVariable* o) {
  middle(o->name << ':' << o->type);
}

void birch::MarkdownGenerator::visit(const MemberVariable* o) {
  middle(o->name << ':' << o->type);
}

void birch::MarkdownGenerator::visit(const Function* o) {
  start("!!! abstract \"");
  middle("function " << o->name);
  if (o->isGeneric()) {
    middle("&lt;" << o->typeParams << "&gt;");
  }
  middle('(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void birch::MarkdownGenerator::visit(const Program* o) {
  line("**program " << o->name << '(' << o->params << ")**\n");
}

void birch::MarkdownGenerator::visit(const MemberFunction* o) {
  start("!!! abstract \"");
  if (o->has(ABSTRACT)) {
    middle("abstract ");
  }
  if (o->has(FINAL)) {
    middle("final ");
  }
  if (o->has(OVERRIDE)) {
    middle("override ");
  }
  middle("function " << o->name);
    if (o->isGeneric()) {
    middle("&lt;" << o->typeParams << "&gt;");
  }
  middle('(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void birch::MarkdownGenerator::visit(const BinaryOperator* o) {
  start("!!! abstract \"");
  middle("operator");
  if (!o->typeParams->isEmpty()) {
    middle('<' << o->typeParams << '>');
  }
  middle(" (" << o->left << ' ' << o->name << ' ' << o->right << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void birch::MarkdownGenerator::visit(const UnaryOperator* o) {
  start("!!! abstract \"");
  middle("operator");
  if (!o->typeParams->isEmpty()) {
    middle('<' << o->typeParams << '>');
  }
  middle(" (" << o->name << ' ' << o->single << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void birch::MarkdownGenerator::visit(const AssignmentOperator* o) {
  auto param = dynamic_cast<const Parameter*>(o->single);\
  assert(param);
  middle(param->type);
}

void birch::MarkdownGenerator::visit(const ConversionOperator* o) {
  middle(o->returnType);
}

void birch::MarkdownGenerator::visit(const SliceOperator* o) {
  start("!!! abstract \"[" << o->params << ']');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void birch::MarkdownGenerator::visit(const Basic* o) {
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

void birch::MarkdownGenerator::visit(const Struct* o) {
  /* lambdas */
  auto docsNotEmpty = [](const Located* o) {
    return !o->loc->doc.empty();
  };
  auto sortByName = [](const Named* o1, const Named* o2) {
    return o1->name->str() < o2->name->str();
  };

  /* anchor for internal links */
  genHead(o->name->str());
  line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");
  start("**struct " << o->name);
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
  finish("**\n\n");
  line(detailed(o->loc->doc) << "\n");

  ++depth;

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

  --depth;
}

void birch::MarkdownGenerator::visit(const Class* o) {
  /* lambdas */
  auto docsNotEmpty = [](const Located* o) {
    return !o->loc->doc.empty();
  };
  auto sortByName = [](const Named* o1, const Named* o2) {
    return o1->name->str() < o2->name->str();
  };

  /* anchor for internal links */
  genHead(o->name->str());
  line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");
  start("**");
  if (o->has(ABSTRACT)) {
    middle("abstract ");
  }
  if (o->has(ACYCLIC)) {
    middle("acyclic ");
  } else if (o->has(FINAL)) {
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
  finish("**\n\n");
  line(detailed(o->loc->doc) << "\n");

  ++depth;

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

  /* slice operators */
  Gatherer<SliceOperator> slices(docsNotEmpty);
  o->accept(&slices);
  if (slices.size() > 0) {
    genHead("Slices");
    line("| Name | Description |");
    line("| --- | --- |");
    ++depth;
    for (auto o : slices) {
      start("| [[...]](#slice) | " << brief(o->loc->doc) << " |");
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

  /* slice operator details */
  if (slices.size() > 0) {
    line("<a name=\"slice\"></a>\n");
    genHead("Member Slice Details");
    ++depth;
    for (auto o : slices) {
      auto desc = quote(detailed(o->loc->doc), "    ");
      *this << o;
      line("");
      line(desc);
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

  --depth;
}

void birch::MarkdownGenerator::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}

void birch::MarkdownGenerator::visit(const NamedType* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle("&lt;" << o->typeArgs << "&gt;");
  }
}

void birch::MarkdownGenerator::visit(const ArrayType* o) {
  middle(o->single << '[');
  for (int i = 0; i < o->depth(); ++i) {
    if (i > 0) {
      middle(',');
    }
    middle("\\_");
  }
  middle(']');
}

void birch::MarkdownGenerator::visit(const TupleType* o) {
  middle('(' << o->single << ')');
}

void birch::MarkdownGenerator::visit(const OptionalType* o) {
  middle(o->single << '?');
}

void birch::MarkdownGenerator::visit(const FutureType* o) {
  middle(o->single << '!');
}

void birch::MarkdownGenerator::visit(const DeducedType* o) {
  //
}

void birch::MarkdownGenerator::genHead(const std::string& name) {
  finish("");
  for (int i = 0; i < depth; ++i) {
    middle('#');
  }
  middle(' ');
  finish(name);
  line("");
}
