from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
from .ddpg_agent import DDPGAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

class DDPGExtension(DDPGAgent):
    def __init__(self, config=None):
        super(DDPGExtension, self).__init__(config)

        self.policy_freq_update = self.cfg.policy_freq_update
        self.iterations = 0

        self.q2 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.q2_target = copy.deepcopy(self.q2)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=float(self.lr))

    def _update(self):
        # get batch data
        batch = self.buffer.sample(self.batch_size, device=self.device)
        # batch contains:
        #    state = batch.state, shape [batch, state_dim]
        #    action = batch.action, shape [batch, action_dim]
        #    next_state = batch.next_state, shape [batch, state_dim]
        #    reward = batch.reward, shape [batch, 1]
        #    not_done = batch.not_done, shape [batch, 1]

        state = batch.state
        action = batch.action
        next_state = batch.next_state
        reward = batch.reward
        not_done = batch.not_done

        # compute current q 
        with torch.no_grad():
            next_q = self.q_target(next_state, self.pi_target(next_state))

        # compute target q
        target_q1 = self.q_target(state, action)
        target_q2 = self.q2_target(state, action)
        target_q = torch.min(target_q1, target_q2)

        target_q = reward + self.gamma * not_done * next_q

        current_q1 = self.q(state, action)
        current_q2 = self.q2(state, action)

        # compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # optimize the critic
        self.q_optim.zero_grad()
        self.q2_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()
        self.q2_optim.step()

        self.iterations+=1

        if self.iterations % self.policy_freq_update == 0:
                
            # compute actor loss
            actor_loss = -torch.mean(self.q(state, self.pi(state)))

            # optimize the actor
            self.pi_optim.zero_grad()
            actor_loss.backward()
            self.pi_optim.step()

            # update the target q and target pi using u.soft_update_params() function
            cu.soft_update_params(self.q, self.q_target, self.tau)
            cu.soft_update_params(self.q2, self.q2_target, self.tau)
            cu.soft_update_params(self.pi, self.pi_target, self.tau)

        return {}
    
    def train_iteration(self):
        #start = time.perf_counter()
        # Run actual training        
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()
        while not done:
            
            # Sample action from policy
            action, _ = self.get_action(obs, evaluation=False) 

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # Add shaped reward
            reward += self.shaped_reward(next_obs)

            # Store action's outcome (so that the agent can improve its policy)        
            
            done_bool = float(done) if timesteps < self.max_episode_steps else 0 
            self.record(obs, action, next_obs, reward, done_bool)
                
            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            
            if timesteps >= self.max_episode_steps:
                done = True
            # update observation
            obs = next_obs.copy()

        # update the policy after one episode
        #s = time.perf_counter()
        info = self.update()
        #e = time.perf_counter()
        
        # Return stats of training
        info.update({
                    'episode_length': timesteps,
                    'ep_reward': reward_sum,
                    })
        
        end = time.perf_counter()
        return info
    
    def shaped_reward(self, state):
        robot = state[0:2]
        num_goals = len(state[2:]) // 2
        goals = state[2:2*(num_goals+1)].reshape(num_goals, 2)
        avoids = state[2*(num_goals+1):].reshape(num_goals, 2)

        # compute distance to closest goal
        dist_to_goal = np.min(np.linalg.norm(goals - robot, axis=1))

        # compute distance to closest avoid
        dist_to_avoid = np.min(np.linalg.norm(avoids - robot, axis=1))

        # compute heuristic reward
        reward = (dist_to_avoid - dist_to_goal) * self.max_action * 0.1

        return reward