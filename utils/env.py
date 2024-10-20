# envs/stock_env.py
from enum import Enum

import pandas as pd
import numpy as np

import gymnasium as gym
from gymnasium import spaces

class Actions(Enum):
    nothing = 0
    long_buy = 1
    short_sell = 2
    cover = 3

class StockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, df, window_size=30):
        self.df = df
        self.window_size = window_size
        self.current_step = window_size
        self.init_balance = 10000 # 初始余额
        self.max_balance = 10000 # 最大资金
        self.balance = 10000 # 余额
        self.drawdown = 1 # 回撤
        self.shares_held = 0 # 持有数量
        self.cost_basis = 0 # 持仓价
        self.held_direction = "" # 持仓方向

        # Define action space
        self.action_space = gym.spaces.Discrete(4)  # buy, sell, nothing

        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, 3),  # 3 features: open, close, mean
            dtype=np.float32
        )

        self._agent_state = np.array([0.0, 0.0, 0.0])

    def _get_obs(self):
        return {"agent": self._agent_state}

    def _get_info(self):
        current_price = self.df.iloc[self.current_step]['Close']
        if self.held_direction == "long":
            self.balance += current_price - self.cost_basis
        elif self.held_direction == "short":
            self.balance += self.cost_basis - current_price
        else:
            pass
        self.max_balance = max(self.max_balance, self.balance)

        self.drawdown = self.max_balance - self.balance
        return {
            "balance": self.balance,
            "drawdown": self.drawdown
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = 10000
        self.max_balance = 10000
        self.cost_basis = 0
        self.shares_held = 0
        self.drawdown = 0
        self.held_direction = "" # long, short, ""
            
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Execute action
        if action == Actions.long_buy:  # buy
            self._long_buy()
        elif action == Actions.cover:  # cover
            self._long_cover()
        elif action == Actions.short_sell:  # sell
            self._short_sell()
        else:  # nothing
            pass

        # Update state
        self.current_step += 1
        obs = self._next_observation()
        info = self._get_info()
        reward = self._calculate_reward()
        done = self._check_if_done()
        truncated = (1 - self.balance/self.init_balance) >= 0.2 # loss of 20%, truncated

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, done, truncated, info

    def _next_observation(self):
        # Get last 30 days of data
        data = self.df.iloc[self.current_step-self.window_size:self.current_step]
        open_values = data['Open'].values
        close_values = data['Close'].values
        mean_values = (open_values + close_values) / 2
        return np.array([open_values, close_values, mean_values]).T

    def _long_buy(self):
        # Buy stock
        if self.balance > 0 and self.shares_held == 0:
            self.shares_held += 1
            self.cost_basis = self.df.iloc[self.current_step]['Close']
            self.held_direction = "long"
    
    def _cover(self):
        # cover stock
        if self.shares_held > 0:
            self.shares_held -= 1
            self.cost_basis = 0
            self.held_direction = ""

    def _short_sell(self):
        # Sell stock
        if self.shares_held == 0:
            self.cost_basis = self.df.iloc[self.current_step]['Close']
            self.shares_held -= 1
            self.held_direction = "short"

    def _calculate_reward(self):
        # Calculate reward based on portfolio value
        portfolio_value = self.balance - self.init_balance
        if self.drawdown == 0:
            self.drawdown += 0.000001
        return portfolio_value / self.drawdown

    def _check_if_done(self):
        # Check if episode is done
        return self.current_step >= len(self.df) - 1

# Example usage
df = pd.read_csv('../data/AAPL.csv')
env = StockTradingEnv(df)
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    blance, drawdown = info['balance'], info['drawdown']
    print(f'Observation: {obs}, Action: {action}, Reward: {reward}, Done: {done}, Blance: {blance}, Drawdown: {drawdown}')