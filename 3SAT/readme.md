# Satisfiability

Given a boolean formula in conjunctive normal form. Does this formula have a satisfying truth assignment ?

 - Literal: A boolean variable or its negation      x or $\bar{x}$
 - Clause: A disjunction(Logical or) of literals                C = $x_1$ v $\bar{x_2}$ v $x_3$

Boolean formula in CNF: A conjunction(Logical and) of clause  $\phi$ = $C_1$ ^ $C_2$ ^ $C_3$

Special case of this satisfiability is
## 3SAT
Sat but each clause has **exactly 3 literals**.\
example formula: ($\bar{x_1}$ v $x_2$ v $x_3$) ^  ($x_1$ v $\bar{x_2}$ v $x_3$) ^  ($x_1$ v $x_2$ v $x_3$) ^  ($\bar{x_1}$ v $\bar{x_2}$ v $\bar{x_3}$)

Value of variables such way that formula becomes true: $x_1$ = true, $x_2$ = true, $x_3$ = false


###  3sat benchmark problems
https://www.cs.ubc.ca/%7Ehoos/SATLIB/benchm.html

https://arxiv.org/pdf/2305.02659