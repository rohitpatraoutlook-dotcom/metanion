from enum import IntEnum
class OpID(IntEnum):
    IDENTITY=1; CONST_ZERO=2; CONST_ONE=3; CONST=4; VAR=5
    ADD=6; SUB=7; MUL=8; DIV=9; POWER=10
    SIN=11; COS=12; TAN=13; EXP=14; LOG=15; LOG10=16
    SQRT=17; SQUARE=18; CUBE=19
    RELU=20; SIGMOID=21; TANH=22
    NEG=23; ABS=24; INVERSE=25; COMPOSE=26
def get_op_name(op):
    names={OpID.IDENTITY:"x", OpID.CONST_ZERO:"0", OpID.CONST_ONE:"1",
           OpID.ADD:"+", OpID.SUB:"-", OpID.MUL:"*", OpID.DIV:"/",
           OpID.SQRT:"sqrt", OpID.EXP:"exp", OpID.LOG:"log", OpID.LOG10:"log10",
           OpID.SIN:"sin", OpID.COS:"cos"}
    return names.get(op, f"op_{op.value}")
def get_op_arity(op):
    if op in {OpID.CONST_ZERO, OpID.CONST_ONE, OpID.CONST, OpID.VAR}: return 0
    unary={OpID.IDENTITY, OpID.NEG, OpID.ABS, OpID.INVERSE, OpID.SQRT, OpID.SQUARE, OpID.CUBE, OpID.EXP, OpID.LOG, OpID.LOG10, OpID.SIN, OpID.COS, OpID.TAN, OpID.RELU, OpID.SIGMOID, OpID.TANH}
    if op in unary: return 1
    return 2
BINARY_OPS=[OpID.ADD, OpID.SUB, OpID.MUL, OpID.DIV]
