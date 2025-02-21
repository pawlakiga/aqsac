# Hybrid Reinforcement Learning for Continuous-Time Industrial Systems with Time-Varying Delays
**Authors**
Iga Pawlak `iga.pawlak@se.abb.com`
Soroush Rastegarpour `soroush.rastegarpour@se.abb.com`
Hamid Feyzmahdavian `hamid.feyzmahdavian@se.abb.com`


The package contains files developed in the scope of the master thesis addressing the problem of delays, particularly unknown continuous and variable ones, in Reinforcement Learning. 

## File structure
- `buffer.py`  - definition of an extended replay buffer used in by safe agents
- `delay_model.py` - delay estimator 
- `delayed-sac.py` - a subclass of SAC to train a delayed agent with model-based prediction. Only used for the comparison of prediction vs augmentation in the report, obsolete in practice. 
- `environment.py` - definition of simulation environments used in the thesis
- `env_wrappers.py` - environments wrappers to rescale action, add delay and augment the state 
- `general_utils.py` - utility functions used in other objects, to scale actions, predict the next states using the model 
- `global_config.py` - definition of default values of variables, such as maximum considered delay or default parameters of the environments
- `dq_sac.py` - definition of the class implementing DQ-SAC
- `aq_sac.py` - definition of the class implementing AQ-SAC
- `orchestrator.py` - definition of the delay-informed agent class, that has the model and un-delayed agent as elements; the transition manager which stores the delay-safe and delay-informed agent and combines their actions; orchestrator 
- `test_utils.py` - utilities for testing
- `train_utils.py` - utilities for training

## Running instructions
### Training
#### Non-delayed agents
The training function for non-delayed agents is defined in `train_utils.py` as `train_default_agent` to be able to easily train the agent with some default parameters and automatic saving of the logs and model. 
The function takes the following arguments
- `core_env` - optional, can be a class inheriting from `gym.Env` that will later be wrapped in appropriate wrappers
- `env_type : str = 'linear'` - passing this argument instead of `core_env` will create the core environemnt inside the function with default parameters. Currently the only accepted values are `linear`, `nonlinear`, `position`, `roborsteer`, `sphericaltank`. 
- `agent_type : str = 'sac'` - agent type, possible other options are `'ddpg'` and `'ppo'`. For `'ddpg'` an OU action noise will be added with `mean = 0` and `sigma = 0.1`.
- `desired_state : float | list = 0.8` - desired state, to be used when using `env_type` and for saving the model and logs. For `LinearVelocity`, `NonLinearVelocity`, `Position` `SphericalTank` can be a float and for `Position` it will be set as desired position. For `RobotSteer` it needs to be a list of two elements for the x position and y position. 
- `n_episodes : int = 100` - number of training episodes
- `ent_coef : float = 0.5` - entropy temperature coefficient for SAC
- `seed : int = None` - seed
- `save : bool = True` - save the model after training 
- `observation_type : str = 'state'` - decides which wrapper to use for observation. Allowed values are `state-error`, `error` and `setpoint-error`. 
- `randomise_setpoint : bool = False` - argument deciding whether to randomise setpoint, useful with `observation_type = 'setpoint-error'` or `observation_type = 'state-error'`. 
- `rescale_action: bool = True` - whether to rescale action between -1 and 1
- `rescale_observation : bool = True` - whether to rescale observation between -1 and 1
- `total_timesteps : int = None` - if not `None`, we train for a number of timesteps instead of a fixed number of episodes

The trained model will be saved in a path 
`'{global_config.MODELS_PATH}/{core_env.unwrapped.__class__.__name__}/{observation_type}/{setpoint}/Desired{desired_state}/{agent_type}'`. 

The logs will be logged to 
`'{global_config.MODELS_PATH}/{core_env.unwrapped.__class__.__name__}/{observation_type}/{setpoint}/Desired{desired_state}/{agent_type}/progress.csv'`. 

### Delay-safe agents and augmented agents
For training delay-safe and augmented agents a `train_default_augmented_agent` function is defined. The function takes all the same arguments as the previous with the differences
- `agent_type : str = 'dqsac'` - defines the type of augmented agent to use. Possible oter arguments are `'aqsac'`, `'augsac'`
- `average_q : bool = True` - whether to use the average or min. If true we'll save the agent as `'dqsac'` and `'aqsac'`, otherwise `'mdqsac'` or `'maqsac'`. 
- `undelayed_critic = None` - undelayed critic for DQ-SAC
- `random_delay : bool = True` - whether to randomise delay for training. Should be true for all safe agents, and false if training the standard augmented SAC for one value of delay. If `True` and `agent_type = 'augsac'` then the agent is saved with as `'drsac'` 
- `init_delay : int = None` - initial delay


**Example loops for training both the non-delayed and safe agents are given in `train-agents.ipynb`. The file also contains code for drawing the training plots using `logs`.**

### Testing
For testing it's always best to use the `test_agent` function from `test_utils.py` with the arguments :
- `deterministic` - applies to SAC, whether to run it deterministically
- `state_error` - specifies if we want to return the observations or if the observation included the error we can also return directly the states of the unwrapped environment. 
- `return_ep_reward` - whether to return the total episode reward

**Example loops for training both the non-delayed and safe agents are given in `test-solution.ipynb`.**
