"""Direct GPU evaluation of Atlas-selected scale dictionaries and gated RMSNorm."""
from contextlib import ExitStack, contextmanager
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def pack_indices(indices, bits):
    rows, groups = indices.shape
    extra = bits - 8
    low = (indices & 255).byte()
    if extra == 0:
        return low
    high = (indices.long() >> 8)
    high = torch.nn.functional.pad(high, (0, (-groups) % 8)).reshape(rows, -1, 8)
    shifts = torch.arange(8, device=indices.device) * extra
    words = (high << shifts).sum(-1)
    byteshifts = 8 * torch.arange(extra, device=indices.device)
    upper = ((words.unsqueeze(-1) >> byteshifts) & 255).byte().reshape(rows, -1)
    return torch.cat((low, upper), dim=1)


def unpack_indices(data, groups, bits):
    j = torch.arange(groups, device=data.device)
    lo = data[:, :groups].int()
    h = bits - 8
    if h == 0:
        return lo
    bitpos = (j % 8) * h
    addr = groups + (j // 8) * h + bitpos // 8
    high = data[:, addr].int()
    if h not in (1, 2, 4, 8):
        high |= data[:, (addr + 1).clamp(max=data.shape[1]-1)].int() << 8
    return lo | (((high >> (bitpos % 8)) & ((1 << h)-1)) << 8)


@triton.jit
def _index(data, row, j, valid, GROUPS: tl.constexpr, BITS: tl.constexpr):
    H: tl.constexpr = BITS - 8
    STRIDE: tl.constexpr = GROUPS + ((GROUPS+7)//8)*H
    start = row*STRIDE
    code = tl.load(data+start+j, mask=valid, other=0).to(tl.int32)
    if H:
        pos = (j%8)*H
        addr = GROUPS+(j//8)*H+pos//8
        upper = tl.load(data+start+addr, mask=valid, other=0).to(tl.int32)
        if H != 1 and H != 2 and H != 4 and H != 8:
            upper |= tl.load(data+start+addr+1, mask=valid & (addr+1<STRIDE),other=0).to(tl.int32)<<8
        code |= (((upper>>(pos%8)) & ((1<<H)-1))<<8)
    return code


@triton.jit
def _reconstruct(q, codes, table, out, numel, K:tl.constexpr,G:tl.constexpr,BITS:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    valid=i<numel
    code=_index(codes,i//K,(i%K)//G,valid,K//G,BITS)
    w=tl.load(q+i,mask=valid,other=0).to(tl.float32)
    s=tl.load(table+code)
    tl.store(out+i,(w*s).to(tl.bfloat16),mask=valid)


def materialize(q, codes, table, group, bits):
    out=torch.empty_like(q,dtype=torch.bfloat16)
    _reconstruct[(triton.cdiv(q.numel(),512),)](q,codes,table,out,q.numel(),q.shape[1],group,bits,512)
    return out


@triton.jit
def _gemv(x,q,codes,table,y,N,K:tl.constexpr,G:tl.constexpr,BITS:tl.constexpr,
          BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)*BN+tl.arange(0,BN)
    valid=row<N
    accum=tl.zeros((BN,),dtype=tl.float32)
    for k in range(0,K,BK):
        for g in tl.static_range(0,BK//G):
            j=k//G+g
            ks=k+g*G+tl.arange(0,G)
            mask=ks<K
            value=tl.load(x+ks,mask=mask,other=0).to(tl.float32)
            w=tl.load(q+row[:,None]*K+ks[None,:],mask=valid[:,None]&mask[None,:],other=0)
            code=_index(codes,row,j,valid & (j<K//G),K//G,BITS)
            scale=tl.load(table+code)
            weight=(w.to(tl.float32)*scale[:,None]).to(tl.bfloat16).to(tl.float32)
            accum+=tl.sum(value[None,:]*weight,axis=1)
    tl.store(y+row,accum.to(tl.bfloat16),mask=valid)


def gemv(x, q, codes, table, group, bits):
    n,k=q.shape
    bk=512 if k%512==0 else 128
    out=x.new_empty((1,1,n))
    _gemv[(triton.cdiv(n,16),)](x.contiguous(),q,codes,table,out,n,k,group,bits,16,bk,num_warps=4)
    return out


def indexed_forward(module):
    def forward(x):
        q,scale=module._atlas_wq,module._atlas_scale
        table,bits=module._region_table,module._region_bits
        if x.shape==(1,1,q.shape[1]) and x.is_cuda and x.dtype==torch.bfloat16 and module.bias is None and not torch.is_grad_enabled():
            return gemv(x,q,scale,table,module._atlas_group,bits)
        weight=materialize(q,scale,table,module._atlas_group,bits)
        return torch.nn.functional.linear(x,weight,module.bias)
    return forward


@triton.jit
def _rms_gate(x,gate,w,out,D:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr,WF32:tl.constexpr):
    row=tl.program_id(0)
    i=tl.arange(0,B)
    pos=row*D+i
    v=tl.load(x+pos,mask=i<D,other=0).to(tl.float32)
    z=tl.load(gate+pos,mask=i<D,other=0).to(tl.float32)
    weight=tl.load(w+i,mask=i<D,other=0).to(tl.float32)
    var=tl.sum(v*v,axis=0)/D
    normalized=(v*libdevice.rsqrt(var+EPS)).to(tl.bfloat16).to(tl.float32)
    scaled=normalized*weight
    if not WF32:
        scaled=scaled.to(tl.bfloat16).to(tl.float32)
    silu=tl.div_rn(z,1.+libdevice.exp(-z))
    tl.store(out+pos,(scaled*silu).to(tl.bfloat16),mask=i<D)


@triton.jit
def _norm_pointwise(x, inv, activated, weight, out, size, D:tl.constexpr, B:tl.constexpr, WF32:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    keep=i<size
    v=tl.load(x+i,mask=keep,other=0).to(tl.float32)
    r=tl.load(inv+i//D,mask=keep,other=0)
    g=tl.load(activated+i,mask=keep,other=0)
    w=tl.load(weight+i%D).to(tl.float32)
    normalized=(v*r).to(tl.bfloat16).to(tl.float32)
    product=normalized*w
    if not WF32:
        product=product.to(tl.bfloat16).to(tl.float32)
    tl.store(out+i,(product*g).to(tl.bfloat16),mask=keep)


def forward_gated_norm(module, x, gate, arithmetic):
    out=torch.empty_like(x)
    width=x.shape[-1]
    if arithmetic=="reference_scalars":
        # Preserve the provider's reduction tree and nonlinear rounding. Only the
        # pointwise BF16-normalize / weight / gate operations share a new launch.
        inv=torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+module.variance_epsilon)
        activated=torch.nn.functional.silu(gate.float())
        _norm_pointwise[(triton.cdiv(x.numel(),256),)](x,inv,activated,module.weight,out,x.numel(),
            width,256,module.weight.dtype==torch.float32,enable_fp_fusion=False)
    elif arithmetic=="fused":
        _rms_gate[(x.shape[0],)](x,gate,module.weight,out,width,module.variance_epsilon,
                                triton.next_power_of_2(width),module.weight.dtype==torch.float32,
                                num_warps=4,enable_fp_fusion=False)
    else:
        raise ValueError(f"unknown gated norm arithmetic {arithmetic}")
    return out


@contextmanager
def bind_gated_norms(model, paths, arithmetic="fused"):
    with ExitStack() as scope:
        for name in paths:
            module=model.get_submodule(name)
            if getattr(module,"activation",None)!="silu" or not hasattr(module,"variance_epsilon"):
                raise ValueError(f"{name} is not the measured SiLU-gated RMSNorm")
            original=module.forward
            def forward(hidden_states, gate, _m=module, _old=original):
                x=hidden_states
                if (not x.is_cuda or torch.is_grad_enabled() or x.dtype!=torch.bfloat16 or x.ndim!=2
                        or x.shape!=gate.shape or not x.is_contiguous() or not gate.is_contiguous()):
                    return _old(x,gate)
                return forward_gated_norm(_m,x,gate,arithmetic)
            if "forward" in vars(module):
                scope.callback(setattr,module,"forward",original)
            else:
                scope.callback(delattr,module,"forward")
            module.forward=forward
        yield
