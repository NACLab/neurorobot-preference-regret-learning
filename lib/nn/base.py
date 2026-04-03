"""
File: base.py
Author: Viet Nguyen
Date: 2024-01-23

"""

from typing import Callable, Tuple, List, Dict
import math
import jax
import jax.numpy as jnp
import numpy as np
# from tensorflow_probability.substrates import jax as tfp
import jax.ad_checkpoint as adc

from . import utils as jaxutils
from . import functional as F
from . import ninjax as nj
from . import derivatives
from . import utils
from . import internal

f32 = jnp.float32
i32 = jnp.int32
# tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
castpd = jaxutils.cast_to_param

def get_dact(name):
  if callable(name):
    return name
  elif name == 'none':
    return derivatives.identity
  elif hasattr(derivatives, name):
    return getattr(derivatives, name)
  else:
    raise NotImplementedError(name)


def get_act(name):
  if callable(name):
    return name
  elif name == 'none' or name == 'identity' or name == None:
    return lambda x: x
  elif name == 'relu':
    # JAX's relu does not have gradient at 0: https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L54-L88
    # f'(x) can have gradient of 1 at 0. https://stackoverflow.com/a/76396054/14861798
    @jax.custom_jvp
    @jax.jit
    def custom_relu(x):
      return jnp.maximum(x, 0)
    custom_relu.defjvps(lambda g, ans, x: jax.lax.select(x >= 0, g, jax.lax.full_like(g, 0)))
    return custom_relu
  elif name == 'gelu_tanh':
    return F.gelu_tanh
  elif name == 'gelu_quick':
    return lambda x: x * jax.nn.sigmoid(1.702 * x)
  elif name == 'relu2':
    return lambda x: jnp.square(jax.nn.relu(x))
  elif name == 'swiglu':
    def fn(x):
      x, y = jnp.split(x, 2, -1)
      return jax.nn.silu(x) * y
    return fn
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise NotImplementedError(name)


def get_initializer(name):
  if callable(name):
    return name
  elif name.endswith(('_in', '_out', '_avg')):
    dist, fan = name.rsplit('_', 1)
  else:
    dist, fan = name, 'in'
  return Initializer(dist, fan, 1.0)


class Initializer:

  def __init__(self, dist='trunc_normal', fan='in', scale=1.0):
    self.dist = dist
    self.fan = fan
    self.scale = scale

  def __call__(self, shape, dtype=jnp.float32, fshape=None):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    assert all(isinstance(x, int) for x in shape), (
      shape, [type(x) for x in shape])
    assert all(x > 0 for x in shape), shape
    fanin, fanout = self.compute_fans(shape if fshape is None else fshape)
    fan = {
      'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout, 'none': 1,
    }[self.fan]
    if self.dist == 'zeros':
      x = jnp.zeros(shape, dtype)
    elif self.dist == 'uniform':
      limit = np.sqrt(1 / fan)
      x = jax.random.uniform(nj.seed(), shape, dtype, -limit, limit)
    elif self.dist == 'normal':
      x = jax.random.normal(nj.seed(), shape)
      x *= np.sqrt(1 / fan)
    elif self.dist == 'trunc_normal':
      x = jax.random.truncated_normal(nj.seed(), -2, 2, shape)
      x *= 1.1368 * np.sqrt(1 / fan)
    elif self.dist == 'normed':
      x = jax.random.uniform(nj.seed(), shape, dtype, -1, 1)
      x *= (1 / jnp.linalg.norm(x.reshape((-1, shape[-1])), 2, 0))
    else:
      raise NotImplementedError(self.dist)
    x *= self.scale
    x = x.astype(dtype)
    return x

  def __repr__(self):
    return f'Initializer({self.dist}, {self.fan}, {self.scale})'

  def __eq__(self, other):
    attributes = ('dist', 'fan', 'scale')
    return all(getattr(self, k) == getattr(other, k) for k in attributes)

  @staticmethod
  def compute_fans(shape):
    if len(shape) == 0:
      return (1, 1)
    elif len(shape) == 1:
      return (1, shape[0])
    elif len(shape) == 2:
      return shape
    else:
      space = math.prod(shape[:-2])
      return (shape[-2] * space, shape[-1] * space)


def get_initializer_pure(name):
  if callable(name):
    return name
  elif name.endswith(('_in', '_out', '_avg')):
    dist, fan = name.rsplit('_', 1)
  else:
    dist, fan = name, 'in'
  return InitializerPure(dist, fan, 1.0)


class InitializerPure:

  def __init__(self, dist='trunc_normal', fan='in', scale=1.0):
    self.dist = dist
    self.fan = fan
    self.scale = scale

  def __call__(self, shape, dtype=jnp.float32, fshape=None):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    assert all(isinstance(x, int) for x in shape), (
      shape, [type(x) for x in shape])
    assert all(x > 0 for x in shape), shape
    fanin, fanout = self.compute_fans(shape if fshape is None else fshape)
    fan = {
      'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout, 'none': 1,
    }[self.fan]
    if self.dist == 'zeros':
      x = jnp.zeros(shape, dtype)
    elif self.dist == 'uniform':
      limit = np.sqrt(1 / fan)
      x = jnp.asarray(np.random.uniform(-limit, limit, shape), dtype)
    elif self.dist == 'normal':
      x = jnp.asarray(np.random.normal(0, 1, shape), dtype)
      x *= np.sqrt(1 / fan)
    elif self.dist == 'normed':
      x = jnp.asarray(np.random.uniform(-1, 1, shape), dtype)
      x *= (1 / jnp.linalg.norm(x.reshape((-1, shape[-1])), 2, 0))
    else:
      raise NotImplementedError(self.dist)
    x *= self.scale
    x = x.astype(dtype)
    return x

  def __repr__(self):
    return f'Initializer({self.dist}, {self.fan}, {self.scale})'

  def __eq__(self, other):
    attributes = ('dist', 'fan', 'scale')
    return all(getattr(self, k) == getattr(other, k) for k in attributes)

  @staticmethod
  def compute_fans(shape):
    if len(shape) == 0:
      return (1, 1)
    elif len(shape) == 1:
      return (1, shape[0])
    elif len(shape) == 2:
      return shape
    else:
      space = math.prod(shape[:-2])
      return (shape[-2] * space, shape[-1] * space)


class Linear(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, units: Tuple[int] | int):
    self.units = (units,) if isinstance(units, int) else tuple(units)

  def macs(self, x):
    # x: (..., input_features)
    insize = x.shape[-1]
    size = math.prod(self.units)
    return int(math.prod(x.shape[:-1]) * insize * size)

  def __call__(self, x):
    jaxutils.ensure_dtypes(x)
    size = math.prod(self.units)
    shape = (x.shape[-1], size)
    x = x @ self.value('kernel', self._scaled_winit, shape).astype(x.dtype)
    if self.bias:
      x += self.value('bias', get_initializer(self.binit), size).astype(x.dtype)
    x = x.reshape((*x.shape[:-1], *self.units))
    return x

  def _scaled_winit(self, *args, **kwargs):
    return get_initializer(self.winit)(*args, **kwargs) * self.outscale


class BlockLinear(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, units, blocks):
    """
    Initialize BlockLinear layer.

    Args:
      units (int): Total number of output units
      blocks (int): Number of blocks to split input and output
    """
    assert isinstance(units, int), (units, type(units))
    assert blocks <= units and units % blocks == 0, (blocks, units)
    self.units = units
    self.blocks = blocks

  def __call__(self, x):
    """
    Apply block linear transformation.

    Args:
      x (jax.Array): Input tensor with shape (..., input_features)
        where input_features must be divisible by self.blocks

    Returns:
      jax.Array: Output tensor with shape (..., self.units)
    """
    jaxutils.ensure_dtypes(x)
    assert x.shape[-1] % self.blocks == 0, (x.shape, self.blocks)
    insize = x.shape[-1]
    shape = (self.blocks, insize // self.blocks, self.units // self.blocks)
    kernel = self.value('kernel', self._scaled_winit, shape).astype(x.dtype)
    x = x.reshape((*x.shape[:-1], self.blocks, insize // self.blocks))
    x = jnp.einsum('...ki,kio->...ko', x, kernel)
    x = x.reshape((*x.shape[:-2], self.units))
    if self.bias:
      x += self.value('bias', get_initializer(self.binit), self.units).astype(x.dtype)
    return x

  def _scaled_winit(self, *args, **kwargs):
    return get_initializer(self.winit)(*args, **kwargs) * self.outscale

class Conv1D(nj.Module):
  # NOTE: TODO: Conv1D with groups have buggy gradient, debugging it

  transp: bool = False
  groups: int = 1
  pad: str = 'same'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = (kernel,) if isinstance(kernel, int) else tuple(kernel)
    self.stride = stride

  def __call__(self, x):
    jaxutils.ensure_dtypes(x)
    # x: (..., length, in_channels)
    # kernel: (kernel, in_channels // groups, out_channels)
    shape = (self.kernel[0], x.shape[-1] // self.groups, self.depth)
    kernel = self.value('kernel', self._scaled_winit, shape).astype(x.dtype)
    if self.transp:
      assert self.pad == 'same', self.pad
      # Manual implementation of fractionally strided convolution (rarely used for 1D)
      x = x.repeat(self.stride, -2)
      mask = ((jnp.arange(x.shape[-2]) - 1) % self.stride == 0)
      x *= mask[:, None]
      stride = (1,)
    else:
      stride = (self.stride,)
    # NHWC for 1D: treat as NLC (batch, length, channels)
    x = jax.lax.conv_general_dilated(
        x, kernel, stride, self.pad.upper(),
        feature_group_count=self.groups,
        dimension_numbers=('NLC', 'LIO', 'NLC'))
    if self.bias:
      x += self.value('bias', get_initializer(self.binit), self.depth).astype(x.dtype)
    return x

  def _scaled_winit(self, *args, **kwargs):
    return get_initializer(self.winit)(*args, **kwargs) * self.outscale


class Conv2D(nj.Module):

  transp: bool = False
  groups: int = 1
  pad: str = 'same'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = (kernel,) * 2 if isinstance(kernel, int) else kernel
    self.stride = stride

  def macs(self, x):
    # x: (B, H, W, C_in)
    B, H, W, C_in = x.shape
    C_out = self.depth
    k_h, k_w = self.kernel
    if self.transp:
      out_h = H * self.stride
      out_w = W * self.stride
    else:
      out_h = math.ceil(H / self.stride)
      out_w = math.ceil(W / self.stride)
    groups = self.groups
    # For grouped convolution the correct total number of multiply-accumulates is:
    # B * out_h * out_w * k_h * k_w * (C_in * C_out) / groups
    # Ensure divisibility for clean integer arithmetic.
    if groups != 1:
      assert C_in % groups == 0 and C_out % groups == 0, (
          f"Channels must be divisible by groups: C_in={C_in}, C_out={C_out}, groups={groups}")
    macs_per_instance = (k_h * k_w) * (C_in * C_out) // groups * out_h * out_w
    total_macs = B * macs_per_instance
    return total_macs

  def __call__(self, x):
    jaxutils.ensure_dtypes(x)
    shape = (*self.kernel, x.shape[-1] // self.groups, self.depth)
    kernel = self.value('kernel', self._scaled_winit, shape).astype(x.dtype)
    if self.transp:
      assert self.pad == 'same', self.pad
      # Manual implementation of fractionally strided convolution because the
      # cuDNN implementation used by XLA has bugs and performance issues.
      x = x.repeat(self.stride, -2).repeat(self.stride, -3)
      maskh = ((jnp.arange(x.shape[-3]) - 1) % self.stride == 0)[:, None]
      maskw = ((jnp.arange(x.shape[-2]) - 1) % self.stride == 0)[None, :]
      x *= (maskh * maskw)[:, :, None]
      stride = (1, 1)
    else:
      stride = (self.stride, self.stride)
    x = jax.lax.conv_general_dilated(
        x, kernel, stride, self.pad.upper(),
        feature_group_count=self.groups,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    if self.bias:
      x += self.value('bias', get_initializer(self.binit), self.depth).astype(x.dtype)
    return x

  def _scaled_winit(self, *args, **kwargs):
    return get_initializer(self.winit)(*args, **kwargs) * self.outscale


class Conv3D(nj.Module):

  transp: bool = False
  groups: int = 1
  pad: str = 'same'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = (kernel,) * 3 if isinstance(kernel, int) else kernel
    self.stride = (stride,) * 3 if isinstance(stride, int) else stride

  def __call__(self, x):
    jaxutils.ensure_dtypes(x)
    if self.transp:
      assert self.groups == 1, self.groups
      shape = (*self.kernel, x.shape[-1], self.depth)
      kernel = self.value('kernel', get_initializer(self.winit), shape).astype(x.dtype)
      x = jax.lax.conv_transpose(
          x, kernel, self.stride, self.pad.upper(),
          dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'))
    else:
      shape = (*self.kernel, x.shape[-1] // self.groups, self.depth)
      kernel = self.value('kernel', get_initializer(self.winit), shape).astype(x.dtype)
      x = jax.lax.conv_general_dilated(
          x, kernel, self.stride, self.pad.upper(),
          feature_group_count=self.groups,
          dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'))
    if self.bias:
      x += self.value('bias', get_initializer(self.binit), self.depth).astype(x.dtype)
    return x


class Norm(nj.Module):

  axis: tuple = (-1,)
  eps: float = 1e-4
  scale: bool = True
  shift: bool = True

  def __init__(self, impl):
    if '1em' in impl:
      impl, exp = impl.split('1em')
      self._fields['eps'] = 10 ** -int(exp)
    self.impl = impl

  def __call__(self, x):
    jaxutils.ensure_dtypes(x)
    dtype = x.dtype
    x = f32(x)
    axis = [a % x.ndim for a in self.axis]
    shape = [x.shape[i] if i in axis else 1 for i in range(min(axis), x.ndim)]
    if self.impl == 'none':
      pass
    elif self.impl == 'rms':
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      scale = self._scale(shape, x.dtype)
      x = x * (jax.lax.rsqrt(mean2 + self.eps) * scale)
    elif self.impl == 'layer':
      mean = x.mean(axis, keepdims=True)
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      var = jnp.maximum(0, mean2 - jnp.square(mean))
      var = adc.checkpoint_name(var, 'small')
      scale = self._scale(shape, x.dtype)
      shift = self._shift(shape, x.dtype)
      x = (x - mean) * (jax.lax.rsqrt(var + self.eps) * scale) + shift
    else:
      raise NotImplementedError(self.impl)
    x = x.astype(dtype)
    return x

  def macs(self, x):
    """Estimate MACs for this normalization on input x.

    This is a heuristic estimate (ops per element) because exact cost depends
    on implementation (reductions, scaling, optional parameters).
    """
    # Total elements in tensor
    try:
      total_elems = int(math.prod(x.shape))
    except Exception:
      # If shape is not available, return 0
      return 0

    # Heuristic ops per element by impl
    if self.impl == 'none':
      ops_per_elem = 0
    elif self.impl == 'rms':
      # square, mean (reduce), add eps, rsqrt, multiply, maybe scale
      ops_per_elem = 5
    elif self.impl == 'layer':
      # mean, mean2, var (sub/square), rsqrt, multiply, shift -> more ops
      ops_per_elem = 10
    else:
      # default conservative estimate
      ops_per_elem = 6

    return int(total_elems * ops_per_elem)

  def _scale(self, shape, dtype):
    if not self.scale:
      return jnp.ones(shape, dtype)
    return self.value('scale', jnp.ones, shape, f32).astype(dtype)

  def _shift(self, shape, dtype):
    if not self.shift:
      return jnp.zeros(shape, dtype)
    return self.value('shift', jnp.zeros, shape, f32).astype(dtype)


class Attention(nj.Module):

  heads: int = 8
  dropout: float = 0.0
  qknorm: str = 'none'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, hidden: int) -> None:
    self.hidden = hidden
    assert self.hidden % self.heads == 0, (self.hidden, self.heads)
    self._hidden = self.hidden // self.heads # hidden per heads
    self._kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)

  def _attention(self, query: jax.Array, key: jax.Array, value: jax.Array,
      mask: jax.Array = None, training: bool = True) -> jax.Array:
    """

    Args:
        query (jax.Array): (B, T, Q)
        key (jax.Array): (B, S1, K)
        value (jax.Array): (B, S2, V)
        mask (jax.Array, optional): (T, S). Defaults to None.
        deterministic (bool, optional): not dropout?. Defaults to False.

    Returns:
        jax.Array: the attention weighted output
    """
    # jaxutils.ensure_dtypes(query)
    # jaxutils.ensure_dtypes(key)
    # jaxutils.ensure_dtypes(value)
    B, T, Q = query.shape
    _query = self.sub("query", Linear, (self.heads, self._hidden), **self._kw)(query).transpose([0, 2, 1, 3]) # (B, H, T, E)
    _key = self.sub("key", Linear, (self.heads, self._hidden), **self._kw)(key).transpose([0, 2, 1, 3]) # (B, H, S1, E)
    _value = self.sub("value", Linear, (self.heads, self._hidden), **self._kw)(value).transpose([0, 2, 1, 3]) # (B, H, S2, E)
    if self.qknorm != 'none':
      _query = self.sub('normq', Norm, self.qknorm)(_query)
      _key = self.sub('normk', Norm, self.qknorm)(_key)
    score = jnp.einsum("BHTE,BHSE->BHTS", _query, _key) / jnp.sqrt(self._hidden)
    score = f32(score) # For doing softmax
    if mask is not None:
      Tq, Tk = _query.shape[2], _key.shape[2]
      assert mask.shape == (B, Tq, Tk), (mask.shape, (B, Tq, Tk))
      _mask = mask.reshape((B, 1, Tq, Tk)) # 'b tq tk -> b 1 tq tk'
      score = F.masked_fill(score, _mask, value=-jnp.inf)
    score = jax.nn.softmax(score, axis=-1)
    score = score.astype(query.dtype)
    score = F.dropout(score, self.dropout, training)
    attention = jnp.einsum("BHTS,BHSE->BHTE", score, _value).transpose([0, 2, 1, 3]).reshape((B, T, -1)) # (B, T, H, E) => (B, T, H, E) => (B, T, E)
    return self.sub("proj", Linear, (Q,), **self._kw, outscale=self.outscale)(attention) # (B, T, Q)

  def __call__(self, x: jax.Array, cond: jax.Array, mask: jax.Array = None, training: bool = True):
    return self._attention(x, cond, cond, mask, training)


class SelfAttentionBlock(nj.Module):

  heads: int = 8
  act: str = 'gelu_tanh'
  dropout: float = 0.0
  qknorm: str = 'none'
  norm: str = 'layer'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, hidden: int):
    self.hidden = hidden
    self._act = get_act(self.act)
    self._attnkw = {k: getattr(self, k) for k in ('heads', 'dropout', 'qknorm', 'bias', 'winit', 'binit', 'outscale')}
    self._mlpkw = {k: getattr(self, k) for k in ('bias', 'winit', 'binit')}
    self._kw = {k: getattr(self, k) for k in ('bias', 'winit', 'binit', 'outscale')}

  def __call__(self, x: jax.Array, mask: jax.Array = None, training: bool = True):
    norm1 = self.sub("ln1", Norm, self.norm)(x) # Normalize before residual pathway
    res1 = self.sub("attn", Attention, self.hidden, **self._attnkw)(norm1, norm1, mask=mask, training=training)
    x = x + res1
    norm2 = self.sub("ln2", Norm, self.norm)(x) # Normalize before residual pathway
    res2 = self.sub("mlp1", Linear, self.hidden, **self._mlpkw)(norm2)
    res2 = self._act(res2)
    res2 = self.sub("mlp2", Linear, self.hidden * 4, **self._mlpkw)(res2)
    res2 = self._act(res2)
    res2 = self.sub("mlp3", Linear, self.hidden, **self._kw)(res2)
    x = x + res2
    return x


class AttentiveProbePooling(nj.Module):

  """Implement the attention pooling mechanism from the JEPA paper from Meta

  Args:
    hidden (int): the hidden dimension of the input
    heads (int): the number of heads
    dropout (float): the dropout rate
    qknorm (str): the normalization type for the query and key
    bias (bool): whether to use bias
    winit (str | Callable): the initializer for the weights
    binit (str | Callable): the initializer for the biases
    outscale (float): the output scale
    act (str): the activation function
    norm (str): the normalization type

  Returns:
    jax.Array: the attention weighted output
  """

  depth: int = 1
  heads: int = 8
  dropout: float = 0.0
  qknorm: str = 'none'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 0.02
  act: str = 'gelu_tanh'
  norm: str = 'layer'

  def __init__(self, hidden: int) -> None:
    self.hidden = hidden
    assert self.hidden % self.heads == 0, (self.hidden, self.heads)
    self._hidden = self.hidden // self.heads # hidden per heads
    self._kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    self._attnkw = {k: getattr(self, k) for k in ('heads', 'dropout', 'qknorm', 'bias', 'winit', 'binit', 'outscale')}
    self._attnblkkw = {k: getattr(self, k) for k in (*self._attnkw, 'act', 'norm')}

  def __call__(self, x: jax.Array, training: bool = True):
    B, T, dim = x.shape
    learnable_query: jax.Array = self.value('learnable_query', get_initializer('zeros'), (1, 1, self.hidden)) # (1, T, dim)
    learnable_query = cast(learnable_query.repeat(B, 0)) # (B, T, dim)
    pooled = self.sub("cross", Attention, self.hidden, **self._attnkw)(learnable_query, x, mask=None, training=training) # (B, 1, hidden)
    for i in range(self.depth):
      pooled = self.sub(f"depth{i}", SelfAttentionBlock, self.hidden, **self._attnblkkw)(pooled, mask=None, training=training) # (B, 1, hidden)
    squeezed = pooled.squeeze(1) # (B, hidden)
    return squeezed

class AttentiveSequenceToSequence(nj.Module):

  """Implement the attention pooling mechanism from the JEPA paper from Meta
  This is the general version of attentive probe pooling where you can provide the length
  of the output sequence.

  Args:
    hidden (int): the hidden dimension of the input
    heads (int): the number of heads
    dropout (float): the dropout rate
    qknorm (str): the normalization type for the query and key
    bias (bool): whether to use bias
    winit (str | Callable): the initializer for the weights
    binit (str | Callable): the initializer for the biases
    outscale (float): the output scale
    act (str): the activation function
    norm (str): the normalization type

  Returns:
    jax.Array: the attention weighted output
  """

  depth: int = 1
  heads: int = 8
  dropout: float = 0.0
  qknorm: str = 'none'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 0.02
  act: str = 'gelu_tanh'
  norm: str = 'layer'

  def __init__(self, hidden: int, target_sequence_length: int) -> None:
    self.hidden = hidden
    self.target_sequence_length = target_sequence_length
    assert self.hidden % self.heads == 0, (self.hidden, self.heads)
    self._hidden = self.hidden // self.heads # hidden per heads
    self._kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    self._attnkw = {k: getattr(self, k) for k in ('heads', 'dropout', 'qknorm', 'bias', 'winit', 'binit', 'outscale')}
    self._attnblkkw = {k: getattr(self, k) for k in (*self._attnkw, 'act', 'norm')}

  def __call__(self, x: jax.Array, training: bool = True):
    B, T, dim = x.shape
    learnable_query: jax.Array = self.value('learnable_query', get_initializer('zeros'), (1, self.target_sequence_length, self.hidden)) # (1, T, dim)
    learnable_query = cast(learnable_query.repeat(B, 0)) # (B, T, dim)
    pooled = self.sub("cross", Attention, self.hidden, **self._attnkw)(learnable_query, x, mask=None, training=training) # (B, T, hidden)
    for i in range(self.depth):
      pooled = self.sub(f"depth{i}", SelfAttentionBlock, self.hidden, **self._attnblkkw)(pooled, mask=None, training=training) # (B, T, hidden)
    return pooled


class MLP(nj.Module):

  act: str = 'silu'
  norm: str = 'rms'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, layers: int = 5, units: int = 1024):
    self.layers = layers
    self.units = units
    self.kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    self.outkw = dict(outscale=self.outscale, **self.kw)

  def __call__(self, x):
    shape = x.shape[:-1]
    x = x.astype(utils.COMPUTE_DTYPE)
    x = x.reshape([-1, x.shape[-1]])
    for i in range(self.layers):
      kw = self.kw if i < self.layers - 1 else self.outkw
      x = self.sub(f'linear{i}', Linear, self.units, **kw)(x)
      x = self.sub(f'norm{i}', Norm, self.norm)(x)
      x = get_act(self.act)(x)
    x = x.reshape((*shape, x.shape[-1]))
    return x

  def macs(self, x):
    # x may have leading batch dims; we reshape similarly to __call__
    try:
      bshape = x.shape[:-1]
      flat_n = int(math.prod(bshape)) if bshape else 1
      in_ch = int(x.shape[-1])
    except Exception:
      return 0

    total = 0
    for i in range(self.layers):
      kw = self.kw if i < self.layers - 1 else self.outkw
      out_units = int(self.units)
      # linear macs
      try:
        lin = Linear(out_units)
        total += int(lin.macs(jnp.zeros((flat_n, in_ch))))
      except Exception:
        total += int(flat_n * in_ch * out_units)
      # norm macs
      try:
        total += int(Norm(self.norm).macs(jnp.zeros((flat_n, out_units))))
      except Exception:
        total += int(flat_n * out_units * 6)
      # activation (~4 ops per element)
      total += int(flat_n * out_units * 4)
      in_ch = out_units

    return int(total)


class GRU(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  norm: str = 'rms'
  update_bias: float = -1.0

  def __init__(self, units: int):
    self.units = units

  def initial(self, batch_size):
    return jnp.zeros((batch_size, self.units), utils.COMPUTE_DTYPE)

  def macs(self, carry, inputs, resets):
    """Compute MACs for GRU operation.
    
    Args:
        carry (jax.Array): (B, U)
        inputs (jax.Array): (B, T, I)
        resets (jax.Array): (B, T)
    
    Returns:
        int: Total number of multiply-accumulate operations
    """
    try:
      B, T, I = inputs.shape
      U = self.units
      
      # Per timestep operations:
      # 1. Concatenation: no MACs
      # 2. Normalization (RMS/Layer): ~6 ops per element for (B, U+I)
      norm_ops = B * (U + I) * 6
      
      # 3. Linear layer: (B, U+I) @ (U+I, 3*U) -> (B, 3*U)
      linear_macs = B * (U + I) * (3 * U)
      
      # 4. Element-wise operations per unit:
      #    - sigmoid(res): ~4 ops per element (U elements)
      #    - multiply (res * cand): 1 MAC per element (U elements)
      #    - tanh(result): ~4 ops per element (U elements)
      #    - sigmoid(update): ~4 ops per element (U elements)
      #    - update * cand: 1 MAC per element (U elements)
      #    - (1 - update): 1 op per element (U elements)
      #    - (1 - update) * carry: 1 MAC per element (U elements)
      #    - final addition: 1 op per element (U elements)
      # Total: ~17 ops per unit
      elementwise_ops = B * U * 17
      
      # Total per timestep
      per_step = norm_ops + linear_macs + elementwise_ops
      
      # Multiply by number of timesteps
      total_macs = int(T * per_step)
      
      return total_macs
    except Exception:
      # Fallback if shape information is not available
      return 0

  def __call__(self, carry, inputs, resets, single=False):
    """_summary_

    Args:
        carry (jax.Array): (B, U)
        inputs (jax.Array): (B, I) or (B, T, I)
        resets (jax.Array): (B, T)
        single (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    assert carry.dtype == utils.COMPUTE_DTYPE, carry.dtype
    assert inputs.dtype == utils.COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype
    if single:
      return self.step(carry, inputs, resets)
    # print(f"[GRU] carry: {carry.shape}, inputs: {inputs.shape}, resets: {resets.shape}")
    carry, outputs = nj.scan(
        lambda carry, args: self.step(carry, *args),
        carry, (inputs, resets), axis=1)
    return carry, outputs

  def step(self, carry, inp, reset):
    # NOTE: When passing previous actions as input, ensure to zero out past
    # actions on is_first and clip actions to bounds if needed.
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    carry = F.mask(carry, ~reset)
    # print(f"[GRU.step] carry: {carry.shape}, inp: {inp.shape}")
    x = jnp.concatenate([carry, inp], -1)
    x = self.sub('norm', Norm, self.norm)(x)
    x = self.sub('linear', Linear, 3 * self.units, **kw)(x)
    res, cand, update = jnp.split(x, 3, -1)
    cand = jnp.tanh(jax.nn.sigmoid(res) * cand)
    update = jax.nn.sigmoid(update + self.update_bias)
    carry = output = update * cand + (1 - update) * carry
    return carry, output


class BidirectionalGRU(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  norm: str = 'rms'
  update_bias: float = -1.0

  def __init__(self, units: int):
    self.units = units

  def initial(self, batch_size):
    # Return initial states for both forward and backward GRUs
    forward_carry = jnp.zeros((batch_size, self.units), utils.COMPUTE_DTYPE)
    backward_carry = jnp.zeros((batch_size, self.units), utils.COMPUTE_DTYPE)
    return jnp.concatenate([forward_carry, backward_carry], axis=-1)

  def macs(self, carry, inputs, resets):
    """Compute MACs for Bidirectional GRU operation.

    Bidirectional GRU processes the sequence in both forward and backward directions,
    so the total MACs is approximately twice that of a single GRU.

    Args:
        carry (jax.Array): (B, 2*U) concatenated forward and backward carries
        inputs (jax.Array): (B, T, I)
        resets (jax.Array): (B, T)

    Returns:
        int: Total number of multiply-accumulate operations
    """
    try:
      # Create a temporary GRU instance to compute MACs for one direction
      forward_gru = GRU(self.units, bias=self.bias, winit=self.winit, binit=self.binit,
        norm=self.norm, update_bias=self.update_bias)

      # Get MACs for forward pass
      B, T, I = inputs.shape
      forward_carry = jnp.zeros((B, self.units), utils.COMPUTE_DTYPE)
      forward_macs = forward_gru.macs(forward_carry, inputs, resets)

      # Backward pass has the same computation cost
      backward_macs = forward_macs

      # Total is sum of both directions
      return int(forward_macs + backward_macs)
    except Exception:
      # Fallback if shape information is not available
      return 0

  def __call__(self, carry, inputs, resets):
    """Bidirectional GRU forward pass

    Args:
        carry (tuple): (forward_carry, backward_carry) each (B, U)
        inputs (jax.Array): (B, T, I)
        resets (jax.Array): (B, T)
        single (bool, optional): single step input. Defaults to False.

    Returns:
        tuple: (carry, outputs) where outputs is concatenated forward and backward outputs
    """
    forward_carry, backward_carry = jnp.split(carry, 2, axis=-1)
    assert forward_carry.dtype == utils.COMPUTE_DTYPE, forward_carry.dtype
    assert backward_carry.dtype == utils.COMPUTE_DTYPE, backward_carry.dtype
    assert inputs.dtype == utils.COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype

    # Forward pass
    forward_carry, forward_outputs = nj.scan(
      lambda carry, args: self.forward_step(carry, *args),
      forward_carry, (inputs, resets), axis=1)

    # Backward pass - reverse the sequence
    reversed_inputs = jnp.flip(inputs, axis=1)
    reversed_resets = jnp.flip(resets, axis=1)

    backward_carry, backward_outputs = nj.scan(
      lambda carry, args: self.backward_step(carry, *args),
      backward_carry, (reversed_inputs, reversed_resets), axis=1)

    # Reverse the backward outputs to match original sequence order
    backward_outputs = jnp.flip(backward_outputs, axis=1)

    # Concatenate forward and backward outputs
    outputs = jnp.concatenate([forward_outputs, backward_outputs], axis=-1)

    return jnp.concatenate([forward_carry, backward_carry], axis=-1), outputs

  def forward_step(self, carry, inp, reset):
    # Forward GRU step
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    carry = F.mask(carry, ~reset)
    x = jnp.concatenate([carry, inp], -1)
    x = self.sub('forward_norm', Norm, self.norm)(x)
    x = self.sub('forward_linear', Linear, 3 * self.units, **kw)(x)
    res, cand, update = jnp.split(x, 3, -1)
    cand = jnp.tanh(jax.nn.sigmoid(res) * cand)
    update = jax.nn.sigmoid(update + self.update_bias)
    carry = output = update * cand + (1 - update) * carry
    return carry, output

  def backward_step(self, carry, inp, reset):
    # Backward GRU step
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    carry = F.mask(carry, ~reset)
    x = jnp.concatenate([carry, inp], -1)
    x = self.sub('backward_norm', Norm, self.norm)(x)
    x = self.sub('backward_linear', Linear, 3 * self.units, **kw)(x)
    res, cand, update = jnp.split(x, 3, -1)
    cand = jnp.tanh(jax.nn.sigmoid(res) * cand)
    update = jax.nn.sigmoid(update + self.update_bias)
    carry = output = update * cand + (1 - update) * carry
    return carry, output


# Operate similar to GRU
class MambaBlock(nj.Module):

  kernel: int = 4
  expand: int = 16 # Expansion factor for input projection => to compute d_inner
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')

  def __init__(self, deter: int, units: int):
    self.units = units # units here is the d_model => or dimension of the input and output vectors of the Mamba block
    self.deter = deter # units here is the d_state => or the dimension of of the state space model's state vector
    self.inners = units * self.expand

  def initial(self, batch_size):
    return jnp.zeros((batch_size, self.inners, self.deter), utils.COMPUTE_DTYPE)

  def _scaled_winit(self, *args, **kwargs):
    return get_initializer(self.winit)(*args, **kwargs)

  def _A_winit(self):
    w = -jnp.arange(1, self.deter + 1)[None].repeat(self.inners, axis=0) # (16D, Z)
    w = jnp.float32(w)
    return w

  def __call__(self, carry, inputs, resets, single=False):
    """_summary_

    Args:
      carry (jax.Array): (B, U)
      inputs (jax.Array): (B, T, I) or (B, I) if single
      resets (jax.Array): (B, T) or (B,) if single
      single (bool, optional): single step input. Defaults to False.

    Returns:
      carry (jax.Array): (B, U)
      outputs (jax.Array): (B, T, U) or (B, U) if single
    """
    ######### Input checks
    assert carry.dtype == utils.COMPUTE_DTYPE, carry.dtype
    assert inputs.dtype == utils.COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype
    # Ensure the inputs are initialized correctly
    x = inputs[:, None] if single else inputs
    assert x.ndim == 3, f"Expected inputs to be 3D, got {x.ndim}D"
    _, seqlen, indims = x.shape

    ########## additional weights initialization needed
    A = self.value('A', self._A_winit).astype(utils.COMPUTE_DTYPE) # (16D, Z)
    D = self.value('D', jnp.ones, (self.inners,)).astype(utils.COMPUTE_DTYPE) # (16D,)

    ########## Main ops
    """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

    Args:
        x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

    Returns:
        output: shape (b, l, d)

    Official Implementation:
        class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
        mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

    """

    # project the inputs to the d_inner * 2 dimension, one used in main, and one used in res
    x = self.sub('inproj', Linear, self.inners * 2, bias=False, winit=self.winit)(x)
    x, res = jnp.split(x, 2, axis=-1) # split into two parts # (B, L, 16D), (B, L, 16D)

    # Extract local feature using a Conv1D
    # x = self.sub('conv', Conv1D, self.inners,
    #   kernel=self.kernel, stride=1, groups=self.inners,
    #   pad='same', winit=self.winit, binit=self.binit)(x) # (B, L, 16D)
    # conv1D with groups have buggy gradient, debugging it
    x = self.sub('local', Linear, self.inners, winit=self.winit, binit=self.binit)(x) # (B, L, 16D)
    x = jax.nn.silu(x) # non-linearity (B, L, 16D)

    ################## SSM Block
    delta_B_C = self.sub('xproj', Linear,
      self.units + 2 * self.deter, bias=False, winit=self.winit)(x) # (B, L, D + 2Z)
    delta, B, C = jnp.split(
      delta_B_C,
      [self.units, self.units + self.deter],
      axis=-1
    ) # delta: (B, L, D). B, C: (B, L, Z)
    delta = self.sub('dtproj', Linear, self.inners, bias=False, winit=self.winit)(delta) # (B, L, 16D)
    delta = jax.nn.softplus(delta) # (B, L, 16D)

    """Does selective scan algorithm. See:
        - Section 2 State Space Models in the Mamba paper [1]
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]

    This is the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

    Args:
        u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
        delta: shape (b, l, d_in)
        A: shape (d_in, n)
        B: shape (b, l, n)
        C: shape (b, l, n)
        D: shape (d_in,)

    Returns:
        output: shape (b, l, d_in)

    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
        Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
    """
    # Selective scan starts here
    # Discretize continuous parameters (A, B)
    # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
    # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
    #   "A is the more important term and the performance doesn't change much with the simplification on B"
    dA = jnp.exp(jnp.einsum('bld,dz->bldz', delta, A))
    dBu = jnp.einsum('bld,blz,bld->bldz', delta, B, x)

    # Selective scan
    # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
    # Note that the below is sequential, while the official implementation does a much faster parallel scan that
    # is additionally hardware-aware (like FlashAttention).
    # carry: (B, 16D, Z), carries: (B, L, 16D, Z), y: (B, L, 16D)
    carry, (carries, y) = nj.scan(
      lambda c, xs: self._step(c, *xs),
      carry, (dA, dBu, C), length=seqlen, axis=1)

    # Add skip connection
    y = y + x * D # (B, L, 16D)

    ############# End of SSM Block

    # Residual connection (again)
    y = y * jax.nn.silu(res)

    # Output projection
    outputs = self.sub('outproj', Linear, self.units, bias=False, winit=self.winit)(y) # (B, L, D)

    if single:
      outputs = outputs[:, 0]
      carries = carries[:, 0]

    return carry, carries, outputs

  def _step(self, carry, dA, dBu, C):
    """
    Single step of the selective scan.
    """
    carry = dA * carry + dBu
    outputs = jnp.einsum('bdz,bz->bd', carry, C)
    return carry, (carry, outputs)  # [batch, *state_dim], ( [batch, *state_dim], [batch, dim] )

class LSTM(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  norm: str = 'rms'
  update_bias: float = -1.0

  def __init__(self, units: int):
    self.units = units

  def initial(self, batch_size):
    return (
      jnp.zeros((batch_size, self.units), utils.COMPUTE_DTYPE), # cell state
      jnp.zeros((batch_size, self.units), utils.COMPUTE_DTYPE) # cell output
    )

  def __call__(self, carry, inputs, resets, single=False):
    """_summary_

    Args:
        carry (jax.Array): ((B, U), (B, U))
        inputs (jax.Array): (B, T, I)
        resets (jax.Array): (B, T)
        single (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    c, h = carry
    assert c.dtype == utils.COMPUTE_DTYPE, c.dtype
    assert h.dtype == utils.COMPUTE_DTYPE, h.dtype
    assert inputs.dtype == utils.COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype
    if single:
      return self.step(carry, inputs, resets)
    carry, outputs = nj.scan(
        lambda carry, args: self.step(carry, *args),
        carry, (inputs, resets), axis=1)
    return carry, outputs

#   def step(self, carry, inp, reset):
#     # NOTE: When passing previous actions as input, ensure to zero out past
#     # actions on is_first and clip actions to bounds if needed.
#     kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
#     cell_state, cell_output = carry
#     cell_state = F.mask(cell_state, ~reset)
#     cell_output = F.mask(cell_output, ~reset)
#     _inp = jnp.concatenate([cell_output, inp], axis=-1)  # (B, I + U)
#     _inp = self.sub('norm', Norm, self.norm)(_inp)
#     # Forget in cell state
#     forget_logit = self.sub('forget', Linear, self.units, **kw)(_inp) # (B, U)
#     forget_probs = jax.nn.sigmoid(forget_logit) # (B, U)
#     cell_state = forget_probs * cell_state
#     # Input in cell state
#     input_proj = self.sub('input_proj', Linear, self.units, **kw)(_inp)  # (B, U)
#     input_proj = jax.nn.tanh(input_proj)  # (B, U)
#     input_logit = self.sub('input_logit', Linear, self.units, **kw)(_inp)  # (B, U)
#     input_probs = jax.nn.sigmoid(input_logit)  # (B, U)
#     cell_state += input_probs * input_proj
#     # Output the cell output (not the state)
#     out_logit = self.sub('output', Linear, self.units, **kw)(_inp)  # (B, U)
#     out_probs = jax.nn.sigmoid(out_logit)  # (B, U)
#     output = jax.nn.tanh(cell_state) * out_probs  # (B, U)
#     carry = (cell_state, output)
#     return carry, output

  # faster
  def step(self, carry, inp, reset):
    # NOTE: When passing previous actions as input, ensure to zero out past
    # actions on is_first and clip actions to bounds if needed.
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    cell_state, cell_output = carry
    cell_state = F.mask(cell_state, ~reset)
    cell_output = F.mask(cell_output, ~reset)
    _inp = jnp.concatenate([cell_output, inp], axis=-1)  # (B, I + U)
    _inp = self.sub('norm', Norm, self.norm)(_inp)
    _inp_out = self.sub('main', Linear, 4 * self.units, **kw)(_inp)
    forget_logit, input_proj, input_logit, out_logit = jnp.split(_inp_out, 4, axis=-1)  # (B, U)
    # Forget in cell state
    forget_probs = jax.nn.sigmoid(forget_logit) # (B, U)
    cell_state = forget_probs * cell_state
    # Input in cell state
    input_proj = jax.nn.tanh(input_proj)  # (B, U)
    input_probs = jax.nn.sigmoid(input_logit)  # (B, U)
    cell_state += input_probs * input_proj
    # Output the cell output (not the state)
    out_probs = jax.nn.sigmoid(out_logit)  # (B, U)
    output = jax.nn.tanh(cell_state) * out_probs  # (B, U)
    carry = (cell_state, output)
    return carry, output


class ConvLSTM(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  norm: str = 'rms'
  update_bias: float = -1.0

  def __init__(self, units: int, kernel: int | tuple):
    self.units = units
    self.kernel = kernel

  def initial(self, batch_size, image_size: tuple):
    width, height = image_size
    return (
      jnp.zeros((batch_size, height, width, self.units), utils.COMPUTE_DTYPE), # cell state
      jnp.zeros((batch_size, height, width, self.units), utils.COMPUTE_DTYPE) # cell output
    )

  def __call__(self, carry, inputs, resets, single=False):
    """_summary_

    Args:
        carry (jax.Array): ((B, H, W, U), (B, H, W, U))
        inputs (jax.Array): (B, T, H, W, I)
        resets (jax.Array): (B, T)
        single (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    c, h = carry
    assert c.dtype == utils.COMPUTE_DTYPE, c.dtype
    assert h.dtype == utils.COMPUTE_DTYPE, h.dtype
    assert inputs.dtype == utils.COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype
    if single:
      return self.step(carry, inputs, resets)
    carry, outputs = nj.scan(
        lambda carry, args: self.step(carry, *args),
        carry, (inputs, resets), axis=1)
    return carry, outputs

  def step(self, carry, inp, reset):
    # NOTE: When passing previous actions as input, ensure to zero out past
    # actions on is_first and clip actions to bounds if needed.
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    cell_state, cell_output = carry
    cell_state = F.mask(cell_state, ~reset)
    cell_output = F.mask(cell_output, ~reset)
    _inp = jnp.concatenate([cell_output, inp], axis=-1)  # (B, I + U)
    _inp = self.sub('norm', Norm, self.norm)(_inp)
    _inp_out = self.sub('main', Conv2D, depth=4 * self.units, kernel=self.kernel, stride=1, pad='same', **kw)(_inp)
    forget_logit, input_proj, input_logit, out_logit = jnp.split(_inp_out, 4, axis=-1)  # (B, U)
    # Forget in cell state
    forget_probs = jax.nn.sigmoid(forget_logit) # (B, U)
    cell_state = forget_probs * cell_state
    # Input in cell state
    input_proj = jax.nn.tanh(input_proj)  # (B, U)
    input_probs = jax.nn.sigmoid(input_logit)  # (B, U)
    cell_state += input_probs * input_proj
    # Output the cell output (not the state)
    out_probs = jax.nn.sigmoid(out_logit)  # (B, U)
    output = jax.nn.tanh(cell_state) * out_probs  # (B, U)
    carry = (cell_state, output)
    return carry, output


class Embedding(nj.Module):

  winit: str | Callable = Initializer('trunc_normal', 'out')

  def __init__(self, classes: int, units: int, shape: Tuple[int] = ()) -> None:
    """

    Args:
        classes (int): _description_
        units (int): _description_
        shape (Tuple[int], optional): event_shape. Defaults to ().
    """
    self.classes = classes
    self.units = units
    self.shape = shape

  def __call__(self, x: jax.Array) -> jax.Array:
    batch_shape = x.shape[:x.ndim - len(self.shape)]
    event_shape = x.shape[x.ndim - len(self.shape):]
    assert event_shape == self.shape, (self.shape, x.shape)
    N = math.prod(self.shape)
    K = self.classes
    D = self.units
    shape = (*self.shape, self.classes, self.units)
    table = self.value('table', get_initializer(self.winit), shape)
    table = table.reshape(N, K, D)
    table = table.astype(utils.COMPUTE_DTYPE)
    index = x.reshape(-1, N)
    embed = table[jnp.arange(N), index]
    embed = embed.reshape(*batch_shape, *self.shape, self.units)
    return embed


class TimeEmbedding(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, units: int, shape: Tuple[int]):
    self.shape = shape # shape needed for displaying timewise e.g., (*sequence_length,)
    self.units = units # dimension of the sinusoidal
    self._kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    self.sin = cast(F.sinusoidal(units, shape))

  def __call__(self) -> jax.Array:
    x = self.sub("in", Linear, self.units, **self._kw)(self.sin) # (*B, dim)
    x = jax.nn.gelu(x) # (*B, dim)
    x = self.sub("out", Linear, self.units, outscale=self.outscale, **self._kw)(x) # (*B, dim)
    return x


class Normalize(nj.Module):

  rate: float = 0.01
  limit: float = 1e-8
  perclo: float = 5.0
  perchi: float = 95.0
  debias: bool = True

  def __init__(self, impl):
    # NOTE: mean_std normalize is very buggy, so we don't use it
    """Keep track of running statistics of the input. Update the statistics
    with the update method and read the statistics with the stats method.

    Args:
        impl (str): 'none', 'meanstd', or 'perc'
        rate (float): the rate of update
        limit (float): the minimum value of the standard deviation
        perclo (float): the lower percentile
        perchi (float): the upper percentile
        debias (bool): whether to debias the statistics
    """
    self.impl = impl
    if self.debias and self.impl != 'none':
      self.corr = nj.Variable(jnp.zeros, (), f32, name='corr')
    if self.impl == 'none':
      pass
    elif self.impl == 'meanstd':
      self.mean = nj.Variable(jnp.zeros, (), f32, name='mean')
      self.sqrs = nj.Variable(jnp.zeros, (), f32, name='sqrs')
    elif self.impl == 'perc':
      self.lo = nj.Variable(jnp.zeros, (), f32, name='lo')
      self.hi = nj.Variable(jnp.zeros, (), f32, name='hi')
    else:
      raise NotImplementedError(self.impl)

  def __call__(self, x, update: bool):
    """Return the statistics of the input.

    Args:
        x (jax.Array): the input to normalize
        update (bool): whether to update the statistics

    Returns:
        jax.Array: the normalized input
    """
    if update:
      self.update(x)
    return self.stats()

  def update(self, x):
    x = sg(f32(x))
    if self.impl == 'none':
      pass
    elif self.impl == 'meanstd':
      self._update(self.mean, self._mean(x))
      self._update(self.sqrs, self._mean(jnp.square(x)))
    elif self.impl == 'perc':
      self._update(self.lo, self._perc(x, self.perclo))
      self._update(self.hi, self._perc(x, self.perchi))
    else:
      raise NotImplementedError(self.impl)
    if self.debias and self.impl != 'none':
      self._update(self.corr, 1.0)

  def stats(self):
    corr = 1.0
    if self.debias and self.impl != 'none':
      corr /= jnp.maximum(self.rate, self.corr.read())
    if self.impl == 'none':
      return 0.0, 1.0
    elif self.impl == 'meanstd':
      mean = self.mean.read() * corr
      std = jnp.sqrt(jax.nn.relu(self.sqrs.read() * corr - mean ** 2))
      std = jnp.maximum(self.limit, std)
      return mean, std
    elif self.impl == 'perc':
      lo, hi = self.lo.read() * corr, self.hi.read() * corr
      return sg(lo), sg(jnp.maximum(self.limit, hi - lo))
    else:
      raise NotImplementedError(self.impl)

  def _mean(self, x):
    x = x.mean()
    axes = internal.get_data_axes()
    if axes:
      x = jax.lax.pmean(x, axes)
    return x

  def _perc(self, x, q):
    axes = internal.get_data_axes()
    if axes:
      x = jax.lax.all_gather(x, axes)
    x = jnp.percentile(x, q)
    return x

  def _update(self, var, x):
    var.write((1 - self.rate) * var.read() + self.rate * sg(x))


class MeanStdNormalize(nj.Module):

  limit: float = 0.1 # min clip for stddev return value

  def __init__(self, use_state: bool = False):
    self.use_state = use_state
    if use_state:
      self._value = self.state_value
      self._write = self.write_state
      self._read = self.read_state
    else:
      self._value = self.params_value
      self._write = self.write_params
      self._read = self.read_params

  def __call__(self, x, update: bool):
    """Return the statistics of the input.

    Args:
        x (jax.Array): the input to normalize
        update (bool): whether to update the statistics

    Returns:
        jax.Array: the normalized input
    """
    if update:
      self.update(x)
    return self.stats()

  def update(self, x):
    old_mu = self._value('mean', jnp.zeros, (x.shape))
    old_std = self._value('stddev', jnp.ones, (x.shape))
    old_skewness = self._value('skewness', jnp.ones, (x.shape))
    count = self._value('count', jnp.zeros, ())
    new_count = count + 1

    new_mu = old_mu + (x - old_mu) / new_count
    new_skewness = old_skewness + (x - old_mu) * (x - new_mu)
    new_std = jnp.sqrt(new_skewness / new_count)

    self._write('mean', sg(new_mu))
    self._write('stddev', sg(new_std))
    self._write('skewness', sg(new_skewness))
    self._write('count', sg(new_count))

  def stats(self):
    mu = self._read('mean')
    std = self._read('stddev')
    return sg(mu), sg(std.clip(self.limit))

class SlowModel:

  def __init__(self, model: nj.Module, *, source: nj.Module, rate: float = 1.0, every: int = 1):
    """Create a model of the same structure as the source model, but with
    parameters updated at a slower rate (target network)
      Normally known as "ema (exponential moving average)" in the literature that is often used in a target network.

    Args:
        model (nj.Module): the model to be updated. Usually the target network.
        source (nj.Module): the source model. Usually the training network.
        rate (float, optional): the rate of update. Defaults to 1.0. Normally, rate is really small like 0.02.
        every (int, optional): the frequency of update. Defaults to 1. Normally, every is 1.
    """
    assert rate == 1 or rate < 0.5, rate
    self.source: nj.Module = source
    self.model: nj.Module = model
    self.rate: float = rate
    self.every: int = every
    name = self.model.name + '_count'
    self.count: nj.Variable = nj.Variable(jnp.zeros, (), i32, name=name)

  def __getattr__(self, name: str):
    self._initonce()
    return getattr(self.model, name)

  def __call__(self, *args, **kwargs):
    self._initonce()
    return self.model(*args, **kwargs)

  def update(self):
    self._initonce()
    mix = jnp.where(self.count.read() % self.every == 0, self.rate, 0)
    fn = lambda src, dst: mix * src + (1 - mix) * dst
    values = jax.tree.map(fn, self.source.values, self.model.values)
    [self.model.write(k, v) for k, v in values.items()]
    self.count.write(self.count.read() + 1)

  def _initonce(self, *args, method=None, **kwargs):
    assert self.source.values, 'no parameters to track'
    if not self.model.values:
      p = self.model.path + '/'
      nj.context().update({p + k: v for k, v in self.source.values.items()})
    assert self.model.values.keys() == self.source.values.keys(), (
        self.model.values.keys(), self.source.values.keys())


class TextEmbedding(nj.Module):

  winit: str | Callable = Initializer('trunc_normal', 'out')

  def __init__(self, hidden: int, vocab: int, seqlen: int):
    self.seqlen = seqlen
    self.token_embedding = Embedding(vocab, hidden, winit=self.winit, name="token_embedding")
    self.position_embedding = Embedding(seqlen, hidden, winit=self.winit, name="position_embedding")

  def __call__(self, tokens: jax.Array) -> jax.Array:
    # tokens: (B, seqlen)
    assert tokens.ndim == 2 and tokens.shape[1] == self.seqlen, tokens.shape
    token_embeddings = self.token_embedding(tokens)
    position_ids = jnp.arange(tokens.shape[-1], dtype=i32)[None, :].repeat(tokens.shape[0], axis=0) # (B, seqlen)
    position_embeddings = self.position_embedding(position_ids)
    return token_embeddings + position_embeddings


