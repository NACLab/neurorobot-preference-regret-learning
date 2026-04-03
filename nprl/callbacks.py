import numpy as np
import re
from typing import Dict
from lib.common.config import Config


#### Record each key value pair of the episode after every episode
def per_episode(ep, mode, logger):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    ep_stats = {
      'length': length,
      'score': score
    }
    if "relative_stability" in ep:
      ep_stats["relative_stability"] = ep["relative_stability"][-1]
    if "success_rate" in ep:
      ep_stats["success_rate"] = ep["success_rate"][-1]
    if "coverage_mean" in ep:
      ep_stats["coverage_mean"] = ep["coverage_mean"][-1]
    # record the number of steps that the agent is in the sucessful states in the episode.
    ep_stats["successfulness"] = np.sum(ep["is_success"].astype(np.float64)) / length
    logger.add(ep_stats, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    return ep


# Add to replay/positive replay everything after observing the whole episode
# TODO: Now this is a batch env, so there can be potential bug
def add_trajectory(ep, replay, positive_replay, worker, use_selfrefl=True):
  T = len(ep["is_first"])
  if ep['has_succeeded'][-1] and use_selfrefl:
    pos_ep = {**ep, "is_positive_memory": np.ones((T,)).astype(np.bool_)}
    [positive_replay.add({k: v[i] for k, v in pos_ep.items() if not k.startswith("log_")}) for i in range(list(pos_ep.values())[0].shape[0])]
  # the normal replay buffer has both positive and negative memories anyway
  norm_ep = {**ep, "is_positive_memory": (np.ones((T,)) * ep['has_succeeded'][-1]).astype(np.bool_)}
  [replay.add({k: v[i] for k, v in norm_ep.items() if not k.startswith("log_")}, worker=worker) for i in range(list(norm_ep.values())[0].shape[0])]
  return ep

def lookback_on_episode(data: Dict[str, np.ndarray], config: Config) -> Dict[str, np.ndarray]:
  """NOTE:
    - when the imitation discount reach toward 1, the agent is treating every state in a
      successful episode its preference, causing future instability
    - when the imitation discount is low enough, the agent only treats a few states in front of
      a successful state as its prior preference => more stable. Especially, when we are doing the expert
      every expert state is treated as prefered state, so it is still good.
    - The same case goes with failure decay rate
  """
  # data: {k: (T, ...)}
  self_imitation_discount = config.self_imitation.disc
  self_imitation_discount_lower_threshold = config.self_imitation.thr
  failure_decay = config.self_imitation.fail
  T = len(data["is_first"])
  data["can_self_imitate"] *= data["has_succeeded"][-1]
  carry = np.maximum(data["can_self_imitate"][-1], 0.0)
  imitation_mask = [carry]
  for t in range(T - 1, 0, -1):
    carry = np.maximum(carry * self_imitation_discount, data["is_success"][t])
    carry *= (carry >= self_imitation_discount_lower_threshold).astype(np.float32)
    imitation_mask.insert(0, carry)
  imitation_mask = np.stack(imitation_mask) # (T)
  final_imitation_mask = np.clip(data["can_self_imitate"] + imitation_mask, 0, 1)
  positive_mask = np.concatenate([
    data["has_succeeded"][:1],
    final_imitation_mask[:-1]
  ], 0)
  failure_decays = np.asarray([failure_decay for i in range(T)]).cumprod(0)[::-1] / failure_decay
  failure_decays *= (1 - data["has_succeeded"][-1])
  neg_rates = -failure_decays
  neg_rates = neg_rates * (1 - data["can_self_imitate"])
  data["can_self_imitate"] = np.asarray(final_imitation_mask, dtype=bool)
  data["pref_label"] = positive_mask + neg_rates
  data["pref_label"][0] = 0
  data["pref_label"] = data["pref_label"].clip(-1, 1) # TODO: Should I only limit to -1 and 1?
  return data


