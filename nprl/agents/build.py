
import sys
from lib.common import print
from lib.common import Config
from lib.envs.build import build_env
from lib.nlp.text_processor import TextProcessor

from .slampc import SLAMPCAgent
from .dummy import DummyAgent


def build_agent(config: Config, text_processor: TextProcessor | None):
  if text_processor is None:
    vocab_size = None
    max_sequence_length = None
  else:
    vocab_size = text_processor.vocab_size
    max_sequence_length = text_processor.max_sequence_length

  # return RAMAgent(config, name='ram')
  env = build_env(config, 0, text_processor=text_processor, multi_task=config.multi_task)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  return {
    'dummy': DummyAgent,
    'slampc': SLAMPCAgent,
  }[config.agent](obs_space, act_space, config=config, vocab_size = vocab_size, max_sequence_length = max_sequence_length)

