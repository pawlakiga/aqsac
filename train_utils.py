import gymnasium as gym

from stable_baselines3 import SAC, DDPG, PPO

from environment import *
from env_wrappers import *
from matplotlib import pyplot as plt
from general_utils import *
# from delay_model import *
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import torch
from aq_sac import *
from stable_baselines3.common.logger import Logger, configure
import datetime
import copy
from test_utils import *

from dq_sac import DQSAC
from delayed_sac import DelayedSAC


def train_default_agent(core_env : gym.Env = None,
                      env_type : str = 'linear', 
                      agent_type : str = 'sac',
                      desired_state : float | list = 0.8, 
                      n_episodes : int = 100, 
                      ent_coef : float = 0.5, 
                      seed : int = None, 
                      save : bool = True, 
                      observation_type : str = 'state',
                      randomise_setpoint : bool = False,
                      rescale_action: bool = True,
                      rescale_observation : bool = True,
                      total_timesteps : int = None):

    if core_env is None:
        core_env = init_core_env(env_type, desired_state, seed) 

    core_env = init_wrappers(core_env, observation_type, randomise_setpoint, rescale_action, rescale_observation)
    env = core_env

    if agent_type == 'sac':
        model = SAC(policy='MlpPolicy', env=env, verbose=1, ent_coef=ent_coef, seed=seed)
    elif agent_type == 'ddpg' : 
        n_actions = core_env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))
        model = DDPG(policy='MlpPolicy', env=env, verbose=1,action_noise=action_noise)
    elif agent_type == 'ppo' :
        model = PPO(policy='MlpPolicy', env=env, verbose=1)
    else:
        raise ValueError("Invalid agent type")

    if randomise_setpoint:
        setpoint = 'randomised'
    else:
        setpoint = 'fixed'

    today = datetime.date.today().strftime("%m%d")
    model_dir = f"{global_config.MODELS_PATH}/{core_env.unwrapped.__class__.__name__}/{observation_type}/{setpoint}/Desired{desired_state}/{agent_type}"
    log_dir = f"{global_config.LOG_PATH}/{core_env.unwrapped.__class__.__name__}/{observation_type}/{setpoint}/{today}/Desired{desired_state}/{agent_type}"
    new_logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)
    if total_timesteps is not None:
        model.learn(total_timesteps = total_timesteps)
    else:
        model.learn(total_timesteps = n_episodes * env.unwrapped.max_episode_len)
    if save:
        model.save(model_dir)
    return model, env

def train_default_augmented_agent(core_env : gym.Env = None,
                                env_type : str = 'linear',
                                agent_type : str = 'dqsac', 
                                average_q : bool = True,
                                undelayed_critic = None, 
                                desired_state : float | list = 0.8, 
                                n_episodes : int = 100, 
                                ent_coef : float = 0.5, 
                                random_delay : bool = True,
                                init_delay : int = None,
                                seed : int = None, 
                                save : bool = True,
                                observation_type : str = 'state', 
                                randomise_setpoint : bool = False, 
                                total_timesteps : int = None,
                                rescale_action: bool = True,
                                rescale_observation : bool = True,):
    # Initialise the environment
    if core_env is None:
        core_env = init_core_env(env_type, desired_state, seed)

    core_env = init_wrappers(core_env, observation_type, randomise_setpoint, rescale_action, rescale_observation)
    # Initial delay
    if init_delay is None or init_delay > global_config.ENV_MAX_DELAY[core_env.unwrapped.__class__.__name__]: 
        init_delay = np.random.randint(0, global_config.ENV_MAX_DELAY[core_env.unwrapped.__class__.__name__])
    # Wrap in delay and augmentation
    env = DelayAction(core_env, delay = init_delay, random_delay=random_delay, max_delay=global_config.ENV_MAX_DELAY[core_env.unwrapped.__class__.__name__])
    env = AugmentState(env, known_delay=global_config.ENV_MAX_DELAY[core_env.unwrapped.__class__.__name__])
    
    if agent_type == 'dqsac':
        safe_model = DQSAC(policy='MlpPolicy', 
                             env=env, 
                             verbose=1, 
                             ent_coef=ent_coef, undelayed_critic=undelayed_critic, avg = average_q)
    elif agent_type == 'aqsac': 
        safe_model = AQSAC(policy='MlpPolicy', 
                                  env=env, 
                                  verbose=1, 
                                  ent_coef=ent_coef, 
                                  avg = average_q)
    elif agent_type == 'drsac': 
        safe_model = SAC(policy='MlpPolicy', env=env, verbose=1, ent_coef=ent_coef)
    else:
        raise ValueError("Invalid agent type")
    
    if not random_delay:
        agent_type = agent_type + 'fixed'

    if average_q == False and agent_type != 'drsac': 
        agent_type = 'm' + agent_type 

    if randomise_setpoint:
        setpoint = 'randomised'
    else:
        setpoint = 'fixed'

    print(f"Env state space {env.observation_space.shape} and action space {env.action_space.shape}")

    today = datetime.date.today().strftime("%m%d")
    model_dir = f"{global_config.MODELS_PATH}/{core_env.unwrapped.__class__.__name__}/{observation_type}/{setpoint}/Desired{desired_state}/{agent_type}"
    log_dir = f"{global_config.LOG_PATH}/{core_env.unwrapped.__class__.__name__}/{observation_type}/{setpoint}/{today}/Desired{desired_state}/{agent_type}"
    new_logger = configure(log_dir, ["stdout", "csv"])
    safe_model.set_logger(new_logger)

    if total_timesteps is not None:
        safe_model.learn(total_timesteps = total_timesteps)
    else :
        safe_model.learn(total_timesteps = n_episodes * env.unwrapped.max_episode_len)
    if save:
        safe_model.save(model_dir)

    # print(f"Delay history is {env.delay_history} and setpoint history {env.desired_states_history}")
    try :
        return safe_model, core_env, env.delay_history, env.desired_states_history
    except :
        try : 
            return safe_model, core_env, env.delay_history, None
        except :
            return safe_model, core_env, None, None
        



def train_default_delayed_sac(core_env : gym.Env = None,
                      env_type : str = 'linear', 
                      agent_type : str = 'sac',
                      init_delay : int = None,
                      desired_state : float | list = 0.8, 
                      n_episodes : int = 100, 
                      ent_coef : float = 0.5, 
                      seed : int = None, 
                      save : bool = True, 
                      observation_type : str = 'state',
                      randomise_setpoint : bool = False,
                      rescale_action: bool = True,
                      rescale_observation : bool = True):

    if core_env is None:
        core_env = init_core_env(env_type, desired_state, seed) 

    core_env = init_wrappers(core_env, observation_type, randomise_setpoint, rescale_action, rescale_observation)

    if init_delay is None or init_delay > global_config.ENV_MAX_DELAY[core_env.unwrapped.__class__.__name__]: 
        init_delay = np.random.randint(0, global_config.ENV_MAX_DELAY[core_env.unwrapped.__class__.__name__])

    env = DelayAction(core_env, delay = init_delay, random_delay=False, return_queue = True)

    if agent_type == 'delayedsac':
        model = DelayedSAC(policy='MlpPolicy', env=env, verbose=1, ent_coef=ent_coef, delay = init_delay)
    else:
        raise ValueError("Invalid agent type")
    
    today = datetime.date.today().strftime("%m%d")
    model_dir = f"{global_config.MODELS_PATH}/{env.unwrapped.__class__.__name__}/Desired{desired_state}/{agent_type}"
    log_dir = f"{global_config.LOG_PATH}/{env.unwrapped.__class__.__name__}/{today}/Desired{desired_state}/{agent_type}"
    new_logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)
    model.learn(total_timesteps = n_episodes * global_config.DEFAULT_PARAMS[env.unwrapped.__class__.__name__]['max_episode_len'])
    if save:
        model.save(model_dir)
    return model, core_env
