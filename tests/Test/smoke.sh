#!/bin/bash
set -eov pipefail

N=4  # for all tests
S=1

eval "`grep -r "program test_basic_" src     | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1/"                         | sort`"
eval "`grep -r "program test_cdf_" src       | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N/"                  | sort`"
eval "`grep -r "program test_grad_" src      | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N --backward false/" | sort`"
eval "`grep -r "program test_grad_" src      | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N --backward true/"  | sort`"
eval "`grep -r "program test_pdf_" src       | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N -S $S --lazy false/"     | sort`"
eval "`grep -r "program test_pdf_" src       | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N -S $S --lazy true/"      | sort`"
eval "`grep -r "program test_z_" src         | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N --lazy false/"     | sort`"
eval "`grep -r "program test_z_" src         | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N --lazy true/"      | sort`"
eval "`grep -r "program test_conjugacy_" src | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N --lazy false/"     | sort`"
eval "`grep -r "program test_conjugacy_" src | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N --lazy true/"      | sort`"
