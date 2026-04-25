import torch
import triton
import triton.language as tl

@triton.jit
def fast_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Convert to base 2
    # log2(e) = 1.4426950408889634
    # ln(2) = 0.6931471805599453
    x_base2 = x * 1.4426950408889634
    
    exp_x = tl.inline_asm_elementwise("ex2.approx.f32 $0, $1;", "=f,f", [x_base2], dtype=tl.float32, is_pure=True, pack=1)
    log2_y = tl.inline_asm_elementwise("lg2.approx.f32 $0, $1;", "=f,f", [y + 1e-6], dtype=tl.float32, is_pure=True, pack=1)
    
    ln_y = log2_y * 0.6931471805599453
    res = exp_x - ln_y

    tl.store(out_ptr + offsets, res, mask=mask)

def test():
    x = torch.randn(1024, device='cuda', dtype=torch.float32)
    y = torch.rand(1024, device='cuda', dtype=torch.float32) + 0.1
    out = torch.empty_like(x)
    
    fast_kernel[(1,)](x, y, out, 1024, 1024)
    
    torch_res = torch.exp(x) - torch.log(y + 1e-6)
    diff = torch.abs(out - torch_res).max().item()
    print("Diff:", diff)

test()
