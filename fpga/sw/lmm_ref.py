"""Integer reference for the LMM accelerator: the contract the RTL must match bit-exactly.

    qx[m,k] = clip(rint(x[m,k] / sx[m]), -Q, Q)   sx[m] = max_k |x[m,k]| / Q   per token
    qw[n,k] = clip(rint(w[n,k] / sw[n]), -Q, Q)   sw[n] = max_k |w[n,k]| / Q   per output channel
    acc     = sum_k qx[m,k] * qw[n,k]              exact; |acc| < K * 2**30 <= 2**40 (K <= 1024)
    y32     = (acc + 2**(S-1)) >> S                arithmetic shift, round half up; |y32| < 2**30
    y       = y32 * 2**S * sx[m] * sw[n]           float32, on the host

Q = 32767 (16-bit, "W16A16"), S = 10. Bits are chosen from fpga/results/e2e_*.json: 8-bit
activations or 8-bit weights flip Laya decisions; 16-bit costs the same Cyclone V DSPs
(native 18x18 multipliers), only memory traffic doubles.

Layout everywhere (host, DDR, RTL): X [M,K] int16 row-major, W [N,K] int16 row-major
(the native nn.Linear weight layout), Y [M,N] int32 row-major, all little-endian.
"""
import numpy as np

SHIFT = 10


def quantize_rows(a, bits=16):
    """Symmetric per-row integer; rint = round-half-to-even, same as C nearbyint()."""
    a = np.asarray(a, dtype=np.float32)
    qmax = np.float32(2 ** (bits - 1) - 1)
    scale = np.abs(a).max(axis=1) / qmax
    scale[scale == 0] = 1  # all-zero row: q = 0, any scale works
    q = np.clip(np.rint(a / scale[:, None]), -qmax, qmax)
    return q.astype(np.int8 if bits <= 8 else np.int16), scale.astype(np.float32)


def int_matmul(qx, qw):
    """acc = qx @ qw.T, exact: |partial sums| < 2**40, far inside float64's 2**53."""
    return np.rint(qx.astype(np.float64) @ qw.astype(np.float64).T).astype(np.int64)


def contract(qx, qw):
    """What the RTL writes to Y: rounded, shifted accumulator as int32."""
    y = (int_matmul(qx, qw) + (1 << (SHIFT - 1))) >> SHIFT
    assert np.abs(y).max(initial=0) < 2**30
    return y.astype(np.int32)


def matmul(x, w_nk, abits=16, wbits=16):
    """FP32 in, FP32 out through the integer contract. x=[M,K], w_nk=[N,K]."""
    qx, sx = quantize_rows(x, abits)
    qw, sw = quantize_rows(w_nk, wbits)
    y = contract(qx, qw).astype(np.float32) * np.float32(2 ** SHIFT)
    return y * sx[:, None] * sw[None, :]


def write_hex64(path, data):
    """Little-endian bytes as one 64-bit word per line, for $readmemh into the TB memory."""
    b = np.ascontiguousarray(data).view(np.uint8).ravel()
    b = np.pad(b, (0, -len(b) % 8))
    with open(path, 'w') as f:
        f.write('\n'.join(f'{v:016x}' for v in b.view('<u8')) + '\n')


def read_hex64(path, count_bytes, dtype):
    words = [int(t, 16) for line in open(path) if not line.startswith('//') for t in line.split()]
    return np.array(words, dtype='<u8').view(np.uint8)[:count_bytes].view(dtype)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    x = rng.standard_normal((9, 768)).astype(np.float32)
    x[3] = 0
    w = rng.standard_normal((64, 768)).astype(np.float32)
    qx, _ = quantize_rows(x)
    assert qx.dtype == np.int16 and qx.min() >= -32767 and (qx[3] == 0).all()
    assert np.rint(np.float32(2.5)) == 2.0                                   # half-to-even
    worst = np.full((1, 1024), 32767, np.int16)
    assert contract(worst, -worst)[0, 0] == (-1024 * 32767**2 + 512) >> SHIFT    # bound edge
    assert (contract(np.array([[1]], np.int16), np.array([[512]], np.int16)) == 1).all()   # 0.5 -> 1
    assert (contract(np.array([[-1]], np.int16), np.array([[512]], np.int16)) == 0).all()  # -0.5 -> 0
    rel = np.linalg.norm(matmul(x, w) - x @ w.T) / np.linalg.norm(x @ w.T)
    assert rel < 1e-3, rel
    print(f'lmm_ref self-check OK, rel err {rel:.2e}')
