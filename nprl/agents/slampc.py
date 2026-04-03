"""
File: slampc.py
Author: Viet Nguyen
Date: 2025-05-27

Description: This contains the slampc agent. Similar to slampc.py but with now learn a value function for preference state as well
"""

import sys
from typing import Dict, Callable, Any, Tuple
import jax
import jax.numpy as jnp
import re
import numpy as np
import collections
from functools import partial as bind
import optax
import chex

from lib import nn
from lib import print
from lib.agent.active import JAXAgent
from lib.common import Space, tree
from lib.nn import distributions

from ..networks import Encoder, Decoder, QueryEncoder, QMoPSSM

# Take space and return if it is an image
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3
# given a dict of dist, sample everything in there
sample = lambda xs: jax.tree.map(lambda x: x.sample(nn.seed()), xs)
# concat a given a list of dicts, concat every child with the same key along axis a
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
# prefix a dict with a given prefix
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}


# NOTE: Working
class SLAMPCAgent(JAXAgent):
  """SLAMPC Agent
  The main neurorobot preference regret learning agent
  Implemented features: functional model predictive control (MPC) scheme
    with state-language-action (SLA) processing

  """
  def __init__(self, obs_space, act_space, config, *, vocab_size: int | None, max_sequence_length: int | None):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space
    self.vocab_size = vocab_size
    self.max_sequence_length = max_sequence_length
    # print(f"act_space: {act_space}", color = "green")
    exclude = ('is_first', 'is_last', 'is_terminal', 'reward', 'is_expert',
      'can_self_imitate', 'is_success', 'has_succeeded', 'success_rate')
    enc_space = {k: v for k, v in obs_space.items() if (k not in exclude and re.match(config.encoder.prockeys, k))}
    print("Encoder space", color='cyan')
    [print(f'  {k:<16} {v}', color='cyan') for k, v in enc_space.items()]
    print()
    dec_space = {k: v for k, v in obs_space.items() if (k not in exclude and re.match(config.decoder.prockeys, k))}
    print("Decoder space", color='green')
    [print(f'  {k:<16} {v}', color='green') for k, v in dec_space.items()]
    print()
    query_space = {k: v for k, v in obs_space.items() if (k not in exclude and re.match(config.query_encoder.prockeys, k))}
    print("Query encoder space", color='cyan')
    [print(f'  {k:<16} {v}', color='cyan') for k, v in query_space.items()]
    print()

    self.feat2tensor = lambda x: jnp.concatenate([
      nn.cast(x['deter']),
      nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))
    ], axis=-1)

    self.encoder = Encoder(enc_space, **config.encoder.simple, name="enc")
    self.query_encoder = QueryEncoder(query_space, vocab_size=vocab_size,
      max_sequence_length=max_sequence_length, **config.query_encoder.simple, name="query_enc")
    self.dynamics = QMoPSSM(act_space, **config.dynamics.qmopssm, name="dyn")
    self.decoder = Decoder(dec_space, **config.decoder.simple, name="dec")

    self.actor = nn.MLPHead(act_space,
      {k: config.policy_dist_disc if v.discrete else config.policy_dist_cont for k, v in act_space.items()},
      **config.actor.dreamer, name="act")

    scalar = Space(np.float32, ())
    binary = Space(bool, (), 0, 1)
    self.rew = nn.MLPHead(scalar, **config.rewhead, name='rew')
    self.slowrew = nn.SlowModel(
        nn.MLPHead(scalar, **config.rewhead, name='slowrew'),
        source=self.rew, **config.slowrew)
    self.con = nn.MLPHead(binary, **config.conhead, name='con')
    self.val = nn.MLPHead(scalar, **config.value, name='val')
    self.slowval = nn.SlowModel(
        nn.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = nn.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = nn.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = nn.Normalize(**config.advnorm, name='advnorm')

    self.modules = [self.encoder, self.query_encoder, self.dynamics, self.decoder,
      self.actor, self.rew, self.con, self.val]

    self.opt = nn.Optimizer(self.modules, **config.opt, name='optimizer')

    scales = self.config.loss_scales.slampc.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    # scales.update({f'pref_{k}': rec for k in dec_space})
    self.scales = scales
    self.actor_refesshing_rate = nn.Variable(nn.f32, self.scales['actor_refreshing'], name='actor_refreshing_rate')
    self.actdisc = {k: v.discrete for k, v in act_space.items()}

  @property
  def policy_keys(self):
    return '.*'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = Space(np.int32)
    spaces['stepid'] = Space(np.uint8, 20)
    spaces['is_expert'] = Space(bool)
    spaces['can_self_imitate'] = Space(bool)
    spaces['is_positive_memory'] = Space(bool)
    spaces['pref_label'] = Space(np.float32)
    if self.config.replay_context:
      spaces.update(tree.flatdict(dict(
          # enc=self.enc.entry_space,
          dyn=self.dynamics.entry_space,
          # dec=self.dec.entry_space
      )))
    return spaces

  def featquery2tensor(self, x):
    out = jnp.concatenate([
      nn.cast(x['deter']),
      nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1))),
      nn.cast(x['prefdeter']),
      nn.cast(x['prefstoch'].reshape((*x['prefstoch'].shape[:-2], -1)))
    ], axis=-1)
    return out

  def pref2feat(self, x):
    return {
      'deter': x['prefdeter'],
      'stoch': x['prefstoch'],
      'logit': x['plogit']
    }

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
      self.dynamics.initial(batch_size),
      jax.tree.map(zeros, self.act_space)
    )

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (dyn_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    tokens = self.encoder(obs, reset, **kw)
    query_tokens = self.query_encoder(obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dynamics.observe(dyn_carry, tokens, prevact, query_tokens, reset, **kw)

    policy = self.actor(self.featquery2tensor(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (dyn_carry, act)
    if self.config.replay_context:
      out.update(tree.flatdict(dict(dyn=dyn_entry,)))
    return carry, act, out


  def train(self, carry, data):
    data = self.populate_data(data)
    carry, obs, prevact, curract, stepid = self._apply_replay_context(carry, data)
    # train
    opt_mets, (carry, entries, outs, mets) = self.opt(
        self._loss, carry, obs, prevact, curract, training=True, has_aux=True)
    mets.update(opt_mets)
    # update slow value
    self.slowval.update()
    self.slowrew.update()
    if self.config.replay_context:
      updates = tree.flatdict(dict(
          stepid=stepid, dyn=entries[0]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # update carry
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, mets


  def _loss(self, carry: Tuple[Dict[str, jax.Array], ...], obs: Dict[str, jax.Array],
      prevact: Dict[str, jax.Array], curract: Dict[str, jax.Array], training: bool):
    mets = {}

    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    (dyn_carry,) = carry

    tokens = self.encoder(obs, reset, training) # (B, T, dim)
    query_tokens = self.query_encoder(obs, reset, training) # (B, T, dim)
    # dyncarry (B, dim), dyn_entries (B, T, dim), los (B, T), repfeat (B, T, dim)
    pref_label = obs['pref_label']
    dyn_carry, dyn_entries, los, repfeat, mets = self.dynamics.loss(dyn_carry, tokens, prevact, query_tokens, pref_label, reset, training)
    losses.update(los)
    metrics.update(mets)
    recons = self.decoder(repfeat, reset, training) # (B, T, dim)
    inp = self.feat2tensor(repfeat) if self.config.reward_grad else nn.sg(self.feat2tensor(repfeat)) # (B, T, dim)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward']) # self.rew(inp, 2): (B, T) => .loss(): (B, T)
    con = nn.f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = nn.f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(nn.sg(target))
    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination # NOTE: Coding this up
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dynamics.starts(dyn_entries, dyn_carry, K) # dyn_entries but with the last K steps
    future_query = self.dynamics.future_query(query_tokens, dyn_carry, K, H) # (B*T, length, dim)
    # feat is the state {deter, stoch, logit, ...}
    policyfn = lambda feat: sample(self.actor(self.featquery2tensor(feat), bdims=1))
    _, imgfeat, imgprevact = self.dynamics.imagine(starts, policyfn, future_query, H, training) # imagine using the dynamics model, imagine already include the preference
    first = jax.tree.map(lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat) # (B*K, 1, dim)
    imgfeat = concat([first if self.config.ac_grads else nn.sg(first), nn.sg(imgfeat)], 1) # (B*K, H+1, dim)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat)) # (B*K, dim)
    lastact = jax.tree.map(lambda x: x[:, None], lastact) # (B*K, 1, dim)
    imgact = concat([imgprevact, lastact], 1) # (B*K, H+1, dim)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    # Imagination loss
    img_wm_ent = self.dynamics._dist(imgfeat['logit']).entropy()
    inp = self.feat2tensor(imgfeat)
    actorinp = self.featquery2tensor(imgfeat)
    rew = self.rew(inp, 2) # (B*K, H+1)
    prefrew = self.rew(self.feat2tensor(self.pref2feat(imgfeat)), 2) # (B*K, H+1)
    regret = prefrew.pred() - rew.pred() # (B*K, H+1)
    los, imgloss_out, mets = imag_loss(
        imgact,
        rew.pred(),
        -regret,
        img_wm_ent,
        self.con(inp, 2).prob(1),
        self.actor(actorinp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    currpolicy = self.actor(self.featquery2tensor(repfeat), bdims=2) # (B, T, dim)
    los, _, mets = actor_refreshing_loss(currpolicy, obs, curract, 1.0, training, self.actdisc, **self.config.actor_refreshing_loss)
    losses.update(los)
    metrics.update(mets)
    # Assertions
    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Replay
    if self.config.repval_loss:
      rep_wm_ent = self.dynamics._dist(repfeat['logit']).entropy()
      # curr_simscore = self.dynamics.cosine_sim(nn.sg(repfeat['prefdeter']), nn.sg(repfeat['prefstoch']), nn.sg(repfeat['deter']), nn.sg(repfeat['stoch']))
      # metrics['simscore_rep'] = curr_simscore.mean()
      feat = nn.sg(repfeat) if self.config.repval_grad else repfeat
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      predact = self.actor(self.featquery2tensor(feat), bdims=2)
      predact = jax.tree.map(lambda x: x.pred(), predact)
      prefrew = self.rew(self.feat2tensor(self.pref2feat(feat)), 2) # (B, K)
      regret = prefrew.pred() - rew
      los, reploss_out, mets = repl_loss(
          last, term, rew, -regret,
          # -difference_between_actor_and_teacher,
          rep_wm_ent, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    # Final loss
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    final_loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    # Outputs and return values
    outs = {'tokens': tokens, 'query_tokens': query_tokens, 'repfeat': repfeat, 'losses': losses}
    next_carry = (dyn_carry,)
    entries = (dyn_entries,)
    return final_loss, (next_carry, entries, outs, metrics)


  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    data = self.populate_data(data)
    carry, obs, prevact, curract, _ = self._apply_replay_context(carry, data)
    (dyn_carry,) = carry

    B, T = obs['is_first'].shape
    # print(f"[_report] B: {B}, T: {T}")
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self._loss(carry, obs, prevact, curract, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self._loss(
              carry, obs, prevact, curract, training=False)[1][2]['losses'][key].mean()
          grad = nn.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dyn_carry, _, obsfeat = self.dynamics.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(firsthalf(outs['query_tokens'])), firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dynamics.imagine(
        dyn_carry, secondhalf(prevact), secondhalf(outs['query_tokens']), length=T - T // 2, training=False)
    obsrecons = self.decoder(obsfeat, firsthalf(obs['is_first']), training=False)
    imgrecons = self.decoder(imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])), training=False)
    prefrecons = self.decoder(self.pref2feat(imgfeat), jnp.zeros_like(secondhalf(obs['is_first'])), training=False)

    # Video preds
    for key in self.decoder.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((nn.i32(pred) - nn.i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      _B, _T, _H, _W, _C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((_T, _H, _B * _W, _C))
      metrics[f'openloop/video/{key}'] = grid

    # Video preferences
    for key in self.decoder.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), prefrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((nn.i32(pred) - nn.i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      _B, _T, _H, _W, _C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((_T, _H, _B * _W, _C))
      metrics[f'openloop/pref/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics


  def _apply_replay_context(self, carry, data):

    (dyn_carry, prevact) = carry # prevact is one previous actions
    carry = (dyn_carry,)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space.keys()}
    ext_obs = {k: data[k] for k in self.ext_space.keys()}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    curract = {k: data[k] for k in self.act_space}
    if not self.config.replay_context:
      return carry, {**obs, **ext_obs}, prevact, curract, stepid

    K = self.config.replay_context
    nested = tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('dyn',)]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
      self.dynamics.truncate(lhs(entries[0]), dyn_carry),
    )
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_ext_obs = {k: rhs(ext_obs[k]) for k in self.ext_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_curract = {k: data[k][:, K:] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, ext_obs, prevact, curract, stepid = jax.tree.map(
      lambda normal, replay: nn.functional.where(first_chunk, replay, normal),
      (carry, rhs(obs), rhs(ext_obs), rhs(prevact), rhs(curract), rhs(stepid)),
      (rep_carry, rep_obs, rep_ext_obs, rep_prevact, rep_curract, rep_stepid))
    return carry, {**obs, **ext_obs}, prevact, curract, stepid


  def populate_data(self, data: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
    # Just for populating data when initialize params
    B, T, *A = data["action"].shape
    if "can_self_imitate" not in data:
      data["can_self_imitate"] = jnp.zeros((B, T), jnp.bool)
    if "is_expert" not in data:
      data["is_expert"] = jnp.zeros((B, T), jnp.bool)
    return data


def imag_loss(act: Dict[str, jax.Array], rew: jax.Array, reward_regret: jax.Array,
    # teacher_actor_sim: jax.Array,
    wment: jax.Array, con: jax.Array,
    policy: Dict[str, distributions.Distribution],
    value: distributions.Distribution, slowvalue: distributions.Distribution,
    retnorm: callable, valnorm: callable, advnorm: callable,
    update: bool, contdisc: bool = True, slowtar: bool = True,
    horizon: int = 333, lam: float = 0.95, actent: float = 3e-4, slowreg: float = 1.0,
    wment_scale: float=1e-4, simscore_scale = 1.0, reward_scale = 1.0,
    teacher_actor_sim_scale = 1.0, reg_scale=0.5, **kwargs
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  internal_reward = reward_scale * rew + reward_regret * reg_scale \
    + wment_scale * wment
  ret = lambda_return(last, term, internal_reward, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(nn.sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}

  policy_loss = nn.sg(weight[:, :-1]) * -(logpi * nn.sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = nn.sg(weight[:, :-1]) * (
      value.loss(nn.sg(tar_padded)) +
      slowreg * value.loss(nn.sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  # metrics['simscore'] = simscore.mean()
  metrics['wment'] = wment.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      # metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)
      metrics[f'rand/{k}'] = ((ents[k].mean() - lo) / (hi - lo)).mean()

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics

def actor_refreshing_loss(policy: Dict[str, distributions.Distribution],
    obs: Dict[str, jax.Array], curract: Dict[str, jax.Array], actor_refreshing_rate: jax.Array,
    training: bool, actdisc: Dict[str, bool], alpha: float = 1.0):
  losses = {}
  metrics = {}
  positive_label = obs['is_expert']
  loss_actor_refreshing = {}
  for k, v in policy.items():
    if actdisc[k]: 
      loss_actor_refreshing[k] = -v.logp(nn.sg(curract[k])) * positive_label
    else: 
      predact = nn.f32(v.sample(seed=nn.seed()))
      loss_actor_refreshing[k] = ((predact - curract[k])**2).sum(-1) * positive_label

  losses['actor_refreshing'] = sum(loss_actor_refreshing.values())
  for k in policy:
    metrics[f'loss_actor_refreshing/{k}'] = loss_actor_refreshing[k].mean()
  return losses, {}, metrics


def repl_loss(
    last, term, rew, reward_regret,
    wment, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
    wment_scale = 1e-4,
    simscore_scale = 1.0,
    reward_scale = 1.0,
    teacher_actor_sim_scale = 1.0,
    reg_scale = 0.5,
    **kwargs
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = nn.f32(~last)
  internal_reward = reward_scale * rew + reward_regret * reg_scale \
    + wment_scale * wment
  ret = lambda_return(last, term, internal_reward, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(nn.sg(ret_padded)) +
      slowreg * value.loss(nn.sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - nn.f32(term))[:, 1:] * disc
  cont = (1 - nn.f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)



