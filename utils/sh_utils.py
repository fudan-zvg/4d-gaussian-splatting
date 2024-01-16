#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   

sh_channels_4d = [1, 6, 16, 33]

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def eval_shfs_4d(deg, deg_t, sh, dirs, dirs_t, l = torch.pi):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    # coeff = (deg + 1) ** 2
    # assert sh.shape[-1] >= coeff

    l0m0 = C0
    result = l0m0 * sh[..., 0]
    
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        l1m1 = -1 * C1 * y
        l1m0 = C1 * z
        l1p1 = -1 * C1 * x
        
        result = (result + 
                  l1m1 * sh[..., 1] +
                  l1m0 * sh[..., 2] +
                  l1p1 * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            
            l2m2 = C2[0] * xy
            l2m1 = C2[1] * yz
            l2m0 = C2[2] * (2.0 * zz - xx - yy)
            l2p1 = C2[3] * xz
            l2p2 = C2[4] * (xx - yy)
            
            result = (result + 
                  l2m2 * sh[..., 4] +
                  l2m1 * sh[..., 5] +
                  l2m0 * sh[..., 6] +
                  l2p1 * sh[..., 7] +
                  l2p2 * sh[..., 8])

            if deg > 2:
                l3m3 = C3[0] * y * (3 * xx - yy)
                l3m2 = C3[1] * xy * z
                l3m1 = C3[2] * y * (4 * zz - xx - yy)
                l3m0 = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                l3p1 = C3[4] * x * (4 * zz - xx - yy)
                l3p2 = C3[5] * z * (xx - yy)
                l3p3 = C3[6] * x * (xx - 3 * yy)
                
                result = (result + 
                  l3m3 * sh[..., 9] +
                  l3m2 * sh[..., 10] +
                  l3m1 * sh[..., 11] +
                  l3m0 * sh[..., 12] +
                  l3p1 * sh[..., 13] +
                  l3p2 * sh[..., 14] +
                  l3p3 * sh[..., 15])
    
    if deg_t > 0:
        t1 = torch.cos(2 * torch.pi * dirs_t / l)
        
        result = (result + 
            t1 * l0m0 * sh[..., 16] +
            t1 * l1m1 * sh[..., 17] +
            t1 * l1m0 * sh[..., 18] +
            t1 * l1p1 * sh[..., 19] + 
            t1 * l2m2 * sh[..., 20] +
            t1 * l2m1 * sh[..., 21] +
            t1 * l2m0 * sh[..., 22] +
            t1 * l2p1 * sh[..., 23] +
            t1 * l2p2 * sh[..., 24] + 
            t1 * l3m3 * sh[..., 25] +
            t1 * l3m2 * sh[..., 26] +
            t1 * l3m1 * sh[..., 27] +
            t1 * l3m0 * sh[..., 28] +
            t1 * l3p1 * sh[..., 29] +
            t1 * l3p2 * sh[..., 30] +
            t1 * l3p3 * sh[..., 31])
        
        if deg_t > 1:
            t2 = torch.cos(2 * torch.pi * 2 * dirs_t / l)
            
            result = (result + 
                t2 * l0m0 * sh[..., 32] +
                t2 * l1m1 * sh[..., 33] +
                t2 * l1m0 * sh[..., 34] +
                t2 * l1p1 * sh[..., 35] + 
                t2 * l2m2 * sh[..., 36] +
                t2 * l2m1 * sh[..., 37] +
                t2 * l2m0 * sh[..., 38] +
                t2 * l2p1 * sh[..., 39] +
                t2 * l2p2 * sh[..., 40] + 
                t2 * l3m3 * sh[..., 41] +
                t2 * l3m2 * sh[..., 42] +
                t2 * l3m1 * sh[..., 43] +
                t2 * l3m0 * sh[..., 44] +
                t2 * l3p1 * sh[..., 45] +
                t2 * l3p2 * sh[..., 46] +
                t2 * l3p3 * sh[..., 47])
                
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5