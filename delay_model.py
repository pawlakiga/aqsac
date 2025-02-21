import numpy as np
import copy
from general_utils import *
import global_config

from scipy.special import softmax 


class IterativeEstimator():
    """Delay estimator using the model error to predict the delay"""
    def __init__(self,
                 state_shape, 
                 max_delay = global_config.MAX_DELAY, 
                 decay : float = 0.99, 
                 scaling_factor : int = 100,
                 useful_axes : list = None) -> None:
        """
        Initialisation
        :param state_shape: Shape of the state of the environment
        :param max_delay: Maximum delay to consider
        :param decay: Decay factor for the model errors to give more importance to new samples
        :param scaling_factor: Scaling factor for the softmax function

        """
        self.max_delay = max_delay
        self.model_errors = np.zeros((max_delay + 1, state_shape))
        self.decay = decay
        self.scaling_factor = scaling_factor
        self.last_dist = np.ones(max_delay + 1) / (max_delay + 1)    
        self.useful_axes = useful_axes

    def update_errors(self, env_model, state, new_state, action_queue = None): 
        """
        Function to update the model errors with the new state
        :param env: Environment, delayed real process, can be augmented
        :param env_model: Environment model
        :param state: Current state
        :param action_queue: Action queue of the delayed environment, should contain also the current action?
        :param new_state: The true new state of the environment
        """
        # If augmented environment
        if state.shape[0] > env_model.observation_space.shape[0]: 
            action_queue = np.concatenate([state[env_model.observation_space.shape[0]:], new_state[-env_model.action_space.shape[0]:]])
            state = state[:env_model.observation_space.shape[0]]
            new_state = new_state[:env_model.observation_space.shape[0]]

        elif action_queue is None:
            # print(f"Action queue is None, cannot update errors")
            return
        # print(f"Action queue is {action_queue}")
        # Reshape the action queue if needed for multi-dimensional actions
        if len(action_queue.shape) == 1 and env_model.action_space.shape[0] > 1: 
            
            action_queue = np.array(action_queue).reshape(action_queue.shape[0]//env_model.action_space.shape[0], env_model.action_space.shape[0])

        new_errors = []
        for delay in range(0, self.max_delay+1): 
            # Predict the next state from delayed action
            # Get old state 
    
            pred_state = predict_next_state(env_model=env_model, cur_state=state, action=action_queue[len(action_queue) - delay - 1])
            new_pred_state = env_model.state
            env_model.reset()
            env_model.set_state(new_state)
            new_true_state = env_model.state
            # print(f"Pred state {pred_state} from {state}")
            # Append the error
            new_errors.append(np.sqrt(np.square(new_true_state - new_pred_state)))
        # print(f"New errors min {np.argmin(new_errors, axis = 0)} are {new_errors}")
        # Update the model errors
        # print(f"Updating errors from {self.model_errors} with shape {self.model_errors.shape}")
        self.model_errors = self.model_errors * self.decay +  np.array(new_errors).reshape(self.model_errors.shape)

    def get_predicted_delay(self): 
        """
        Function to get the predicted delay
        """
        probs = self.get_probs()
        # return np.random.choice(range(self.max_delay+1), p = probs)
        # return np.argmin(self.model_errors)
        # print(f"Pred delay is {np.argmax(probs)} with probs {probs}")
        return np.argmax(probs)

    def get_certainty(self):
        """
        Certainty of the delay prediction, calculated as the max probability
        """
        # If we have multiple dimensions, use sum 
        probs = self.get_probs()

        max_prob = np.max(probs)
        
        return max_prob
    
    def get_probs(self): 
        # If we have multiple dimensions, use sum 
        if len(self.model_errors.shape) > 1 : 
            if self.useful_axes is not None: 
                probs = softmax(-self.scaling_factor * np.sum(self.model_errors[:,self.useful_axes], axis = -1))
            else:
                probs = softmax(-self.scaling_factor * np.sum(self.model_errors, axis = -1))
        else :
            probs = softmax(-self.scaling_factor * self.model_errors)

        return probs
    
    def predict(self, env): 
        """
        Function to predict the delay and certainty
        :param env: Environment, delayed real process
        """
        return self.get_predicted_delay(env), self.get_certainty()

    def reset(self): 
        """
        Function to reset the model errors
        """
        self.model_errors = np.zeros((self.max_delay + 1, self.model_errors.shape[-1]))
        self.last_dist = np.ones(self.max_delay + 1) / (self.max_delay + 1)
