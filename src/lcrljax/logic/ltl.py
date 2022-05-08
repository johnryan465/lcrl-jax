from dataclasses import dataclass
from typing import Literal, Union
from lcrljax.automata.ldba import LDBA

from pyparsing import infixNotation, opAssoc, Keyword, Word, alphas, ParserElement

ParserElement.enablePackrat()


class BaseOperand:
    def __init__(self, t):
        self.label = t[0]

    def __str__(self) -> str:
        return str(self.label)


class Variable(BaseOperand):
    def __init__(self, t):
        super().__init__(t)


class TrueOperand(BaseOperand):
    def __init__(self, t):
        super().__init__(t)


class BaseOperator:
    symbol: str
    pre: int
    assoc: Union[Literal[opAssoc.LEFT], Literal[opAssoc.RIGHT]]


class BoolMonOp(BaseOperator):
    operand: "LTL"

    def __init__(self, t):
        self.arg = t[0][1]

    def __str__(self) -> str:
        return self.symbol + str(self.arg)


class BoolBinOp(BaseOperator):

    def __init__(self, t):
        self.args = t[0][0::2]

    def __str__(self) -> str:
        sep = " %s " % self.symbol
        return "(" + sep.join(map(str, self.args)) + ")"

    @property
    def left(self):
        return self.args[0]

    @property
    def right(self):
        return self.args[1]


class LNOT(BoolMonOp):
    symbol = "!"
    pre = 1
    assoc = opAssoc.RIGHT


class Next(BoolMonOp):
    symbol = "X"
    pre = 1
    assoc = opAssoc.RIGHT


class Until(BoolBinOp):
    symbol = "U"
    pre = 2
    assoc = opAssoc.LEFT


class LAND(BoolBinOp):
    symbol = "&"
    pre = 2
    assoc = opAssoc.LEFT


class LOR(BoolBinOp):
    symbol = "|"
    pre = 2
    assoc = opAssoc.LEFT


# define keywords and simple infix notation grammar for boolean
# expressions
LTL = Union[TrueOperand, LAND, LNOT, Next, Until, Variable]

OPS = [LNOT, Next, Until, LAND, LOR]

TRUE = Keyword("True")
TRUE.setParseAction(TrueOperand)

VAR = Word(alphas, max=1)
VAR.setParseAction(Variable)

boolOperand = TRUE | VAR

# define expression, based on expression operand and
# list of operations in precedence order
boolExpr = infixNotation(boolOperand, [(Keyword(op.symbol), op.pre, op.assoc, op)
                         for op in OPS]).setName("boolean_expression")


class LinearTemporalLogic:
    def __init__(self, formula: LTL):
        self.formula = formula

    @classmethod
    def from_file(cls, filename):
        """
        Reads a LTL formula from a file.
        """
        with open(filename, 'r') as f:
            return cls.from_string(f.read())

    @classmethod
    def from_string(cls, string: str):
        """
        Reads a LTL formula from a string.
        """
        return cls(boolExpr.parseString(string)[0])

    def as_ldba(self) -> LDBA:
        """
        Returns an automata that accepts the LTL formula.
        """
        pass

    @classmethod
    def _to_string(cls, formula: LTL) -> str:
        """
        Returns a string representation of the LTL formula.
        """
        if isinstance(formula, TrueOperand):
            return "True"
        elif isinstance(formula, LAND):
            return "({} & {})".format(
                LinearTemporalLogic._to_string(formula.left),
                LinearTemporalLogic._to_string(formula.right))
        elif isinstance(formula, LNOT):
            return "!{}".format(LinearTemporalLogic._to_string(formula.operand))
        elif isinstance(formula, Next):
            return "X{}".format(LinearTemporalLogic._to_string(formula.operand))
        elif isinstance(formula, Until):
            return "({} U {})".format(
                LinearTemporalLogic._to_string(formula.left),
                LinearTemporalLogic._to_string(formula.right))
        elif isinstance(formula, Variable):
            return formula.label

    def __str__(self) -> str:
        return LinearTemporalLogic._to_string(self.formula)


if __name__ == "__main__":
    ltl = LinearTemporalLogic.from_file("1.txt")
    print(ltl)
