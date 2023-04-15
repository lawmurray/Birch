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

  /* gather items of interest */
  auto all = [](const void*) { return true; };
  auto docsNotEmpty = [](const Located* o) { return !o->loc->doc.empty(); };

  Gatherer<GlobalVariable> variables(docsNotEmpty);
  Gatherer<Program> programs(docsNotEmpty);
  Gatherer<Function> functions(docsNotEmpty);
  Gatherer<BinaryOperator> binaries(all);
  Gatherer<UnaryOperator> unaries(all);
  Gatherer<Basic> basics(docsNotEmpty);
  Gatherer<Struct> structs(docsNotEmpty);
  Gatherer<Class> classes(docsNotEmpty);

  o->accept(&variables);
  o->accept(&programs);
  o->accept(&functions);
  o->accept(&binaries);
  o->accept(&unaries);
  o->accept(&basics);
  o->accept(&structs);
  o->accept(&classes);

  /* alphabetical sorting of items within categories */
  auto sortByName = [](const Named* o1, const Named* o2) {
    return o1->name->str() < o2->name->str();
  };
  std::stable_sort(variables.begin(), variables.end(), sortByName);
  std::stable_sort(programs.begin(), programs.end(), sortByName);
  std::stable_sort(functions.begin(), functions.end(), sortByName);
  std::stable_sort(binaries.begin(), binaries.end(), sortByName);
  std::stable_sort(unaries.begin(), unaries.end(), sortByName);
  std::stable_sort(basics.begin(), basics.end(), sortByName);
  std::stable_sort(structs.begin(), structs.end(), sortByName);
  std::stable_sort(classes.begin(), classes.end(), sortByName);

  /* load up type names, these are used for automatic linking */
  std::for_each(basics.begin(), basics.end(), [this](const Basic* o) {
        basicNames.insert(o->name->str());
      });
  std::for_each(structs.begin(), structs.end(), [this](const Struct* o) {
        structNames.insert(o->name->str());
      });
  std::for_each(classes.begin(), classes.end(), [this](const Class* o) {
        classNames.insert(o->name->str());
      });

  /* global variables */
  if (variables.size() > 0) {
    line(head("Variables"));
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
  if (programs.size() > 0) {
    line(head("Programs"));
    ++depth;
    std::string desc;
    for (auto o : programs) {
      line(head(o->name->str()));
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
  if (functions.size() > 0) {
    line(head("Functions"));
    ++depth;
    std::string name, desc;
    for (auto o : functions) {
      if (o->name->str() != name) {
        /* heading only for the first overload of this name */
        line(head(o->name->str()));
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
  if (binaries.size() + unaries.size() > 0) {
    line(head("Operators"));
    ++depth;
    std::string name, desc;

    for (auto o : binaries) {
      if (o->name->str() != name) {
        name = o->name->str();

        /* heading only for the first overload of this name */
        if (name == "+" || name == "-") {
          line(head(name + " (binary)"));
        } else {
          line(head(name));
        }
        line("<a name=\"" << anchor(name) << "\"></a>\n");
      }
      desc = quote(detailed(o->loc->doc), "    ");
      *this << o;
      line("");
      line(desc);
      line("");
    }
    line("");

    for (auto o : unaries) {
      if (o->name->str() != name) {
        name = o->name->str();

        /* heading only for the first overload of this name */
        if (name == "+" || name == "-") {
          line(head(name + " (unary)"));
        } else {
          line(head(name));
        }
        line("<a name=\"" << anchor(name) << "\"></a>\n");
      }
      desc = quote(detailed(o->loc->doc), "    ");
      *this << o;
      line("");
      line(desc);
      line("");
    }
    line("");
    
    --depth;
  }

  /* basic types */
  line(head("Types"));
  ++depth;
  for (auto o : basics) {
    *this << o;
  }
  --depth;

  /* structs */
  line(head("Structs"));
  ++depth;
  for (auto o : structs) {
    *this << o;
  }
  --depth;

  /* classes */
  line(head("Classes"));
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
  middle(" (" << o->name << o->single << ')');
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
  /* anchor for internal links */
  line(head(o->name->str()));
  line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");
  start("**type " << o->name);
  if (o->isGeneric()) {
    middle("&lt;" << o->typeParams << "&gt;");
  }
  if (!o->base->isEmpty()) {
    if (o->isAlias()) {
      middle(" = ");
    } else {
      middle(" < ");
    }
    middle(o->base);
  }
  finish("**\n\n");
  line(detailed(o->loc->doc) << "\n");
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
  line(head(o->name->str()));
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
    line(head("Member Variables"));
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
  line(head(o->name->str()));
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
    line(head("Assignments"));
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
    line(head("Conversions"));
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
    line(head("Slices"));
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
    line(head("Member Variables"));
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
    line(head("Member Functions"));
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
    line(head("Member Slice Details"));
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
    line(head("Member Function Details"));
    ++depth;
    std::string name, desc;
    for (auto o : functions) {
      if (o->name->str() != name) {
        /* heading only for the first overload of this name */
        line(head(o->name->str()));
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
  if (basicNames.find(o->name->str()) != basicNames.end()) {
    middle('[' << o->name << "](../../types/" << o->name << ')');
  } else if (structNames.find(o->name->str()) != structNames.end()) {
    middle('[' << o->name << "](../../structs/" << o->name << ')');
  } else if (classNames.find(o->name->str()) != classNames.end()) {
    middle('[' << o->name << "](../../classes/" << o->name << ')');
  } else {
    middle(o->name);
  }
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

std::string birch::MarkdownGenerator::head(const std::string& name) {
  std::stringstream buf;
  buf << '\n';
  for (int i = 0; i < depth; ++i) {
    buf << '#';
  }
  buf << ' ' << name << "\n\n";
  return buf.str();
}

std::string birch::MarkdownGenerator::detailed(const std::string& str) {
  static const std::string name("\\b[αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩA-Za-z0-9_]+\\b");
  static const std::regex newline(" *\n *\\* ?");
  static const std::regex param("@t?param *(" + name + ')');
  static const std::regex p("@p *(" + name + ')');
  static const std::regex ret("@return");
  static const std::regex see("@see");
  static const std::regex admonition("@(attention|bug|example|note|quote|todo|warning)");

  std::string r = str;
  r = std::regex_replace(r, newline, "\n");
  r = std::regex_replace(r, param, "  - **$1** ");
  r = std::regex_replace(r, p, "**$1**");
  r = std::regex_replace(r, ret, "**Returns** ");
  r = std::regex_replace(r, see, "**See also** ");
  r = std::regex_replace(r, admonition, "!!! $1");
  return r;
}

std::string birch::MarkdownGenerator::brief(const std::string& str) {
  std::string str1(one_line(str));
  std::smatch match;
  std::regex reg(".*?[\\.\\?\\!]");
  if (std::regex_search(str1, match, reg)) {
    return one_line(match.str());
  } else {
    return "";
  }
}

std::string birch::MarkdownGenerator::one_line(const std::string& str) {
  return std::regex_replace(detailed(str), std::regex("\\n"), " ");
}

std::string birch::MarkdownGenerator::anchor(const std::string& name) {
  return std::regex_replace(lower(name), std::regex(" |_"), "-");
}

std::string birch::MarkdownGenerator::quote(const std::string& str,
    const std::string& indent) {
  return std::regex_replace(indent + str, std::regex("\\n"),
      std::string("\n") + indent);
}
