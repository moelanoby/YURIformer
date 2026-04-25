import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# POSIT16 ARITHMETIC KERNELS (TRITON)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def posit_op_kernel(
    a_ptr, b_ptr, out_ptr, 
    n_elements, 
    op_type: tl.constexpr, # 0: add, 1: sub, 2: mul, 3: div
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Convert from Posit16 (log-space) to Float for arithmetic
    s_a = tl.where(a >= 0, 1.0, -1.0)
    af = s_a * (tl.exp(tl.abs(a)) - 1.0)
    
    s_b = tl.where(b >= 0, 1.0, -1.0)
    bf = s_b * (tl.exp(tl.abs(b)) - 1.0)

    if op_type == 0: res_f = af + bf
    elif op_type == 1: res_f = af - bf
    elif op_type == 2: res_f = af * bf
    else: res_f = af / (bf + 1e-9)

    # Convert from Float to Posit16 (log-space)
    s_res = tl.where(res_f >= 0, 1.0, -1.0)
    res = s_res * tl.log(1.0 + tl.abs(res_f))
    
    tl.store(out_ptr + offsets, res, mask=mask)

# ─────────────────────────────────────────────────────────────────────────────
# POSIT16 DATATYPE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Posit16:
    """
    Efficient Posit16 Datatype Emulation.
    Stores values in a compressed logarithmic space to mimic Posit16's 
    dynamic range and precision profile.
    """
    def __init__(self, tensor, is_compressed=False):
        if is_compressed:
            self.data = tensor
        else:
            self.data = self.encode(tensor)

    @staticmethod
    def encode(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def decode(x):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

    def to_float(self):
        return self.decode(self.data)

    def _apply_op(self, other, op_type):
        if isinstance(other, Posit16):
            b_data = other.data
        else:
            b_data = self.encode(other)
            
        out = torch.empty_like(self.data)
        n = self.data.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        posit_op_kernel[grid](self.data, b_data, out, n, op_type, BLOCK_SIZE=1024)
        return Posit16(out, is_compressed=True)

    def __add__(self, other): return self._apply_op(other, 0)
    def __sub__(self, other): return self._apply_op(other, 1)
    def __mul__(self, other): return self._apply_op(other, 2)
    def __truediv__(self, other): return self._apply_op(other, 3)
    
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return Posit16(other) - self
    def __rmul__(self, other): return self * other

    def __pow__(self, p): 
        if p == 2: return self * self
        return Posit16(torch.pow(self.to_float(), p))

    def __repr__(self):
        return f"Posit16({self.to_float()})"

    @property
    def shape(self): return self.data.shape
    def norm(self): return torch.norm(self.to_float())
