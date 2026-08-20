from .op_enum import OpID, get_op_arity
class HashConsingPool:
    def __init__(self):
        self._pool, self._reverse, self._next_handle = {}, {}, 1
        for op, val in [(OpID.IDENTITY,1),(OpID.CONST_ZERO,2),(OpID.CONST_ONE,3)]:
            self._pool[(op,None,None)] = val; self._reverse[val]=(op,None,None)
        self._next_handle = 4
    def intern(self, op, left=None, right=None, value=None, index=None):
        if op == OpID.CONST:
            if value is None: raise ValueError("CONST needs value")
            key=(op,value)
        elif op == OpID.VAR:
            if index is None: raise ValueError("VAR needs index")
            key=(op,index)
        elif get_op_arity(op) == 0:
            key=(op,None,None)
        else:
            key=(op,left,right)
        if key in self._pool: return self._pool[key]
        h=self._next_handle; self._next_handle+=1
        self._pool[key]=h; self._reverse[h]=key
        return h
    def get_node(self, h): return self._reverse.get(h)
_POOL=None
def get_pool():
    global _POOL
    if _POOL is None: _POOL=HashConsingPool()
    return _POOL
def reset_pool():
    global _POOL; _POOL=None
def intern(op,left=None,right=None,value=None,index=None): return get_pool().intern(op,left,right,value,index)
def lookup(h): return get_pool().get_node(h)
