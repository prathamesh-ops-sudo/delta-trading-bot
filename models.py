"""
AI/ML Models Module for Forex Trading System
Includes LSTM, Transformer, and DQN Reinforcement Learning models
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import random

import numpy as np
import pandas as pd

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    F = None
    DataLoader = None
    TensorDataset = None
    print("PyTorch not available")
    
    class _DummyModule:
        pass
    
    class _DummyNN:
        Module = _DummyModule
        Linear = None
        Dropout = None
        ReLU = None
        Sequential = None
        MSELoss = None
        CrossEntropyLoss = None
        TransformerEncoderLayer = None
        TransformerEncoder = None
    
    nn = _DummyNN()

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import config

logger = logging.getLogger(__name__)


# ==================== LSTM Model (TensorFlow) ====================

class LSTMPricePredictor:
    """LSTM model for price prediction"""
    
    def __init__(self, input_shape: Tuple[int, int] = None,
                 hidden_units: int = None, num_layers: int = None,
                 dropout: float = None, output_size: int = 3):
        
        self.input_shape = input_shape or (config.ml.lstm_sequence_length, 50)
        self.hidden_units = hidden_units or config.ml.lstm_hidden_units
        self.num_layers = num_layers or config.ml.lstm_layers
        self.dropout = dropout or config.ml.lstm_dropout
        self.output_size = output_size
        
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
        if TF_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """Build LSTM model architecture"""
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        
        # LSTM layers
        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            x = layers.LSTM(
                self.hidden_units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.dropout / 2
            )(x)
            x = layers.BatchNormalization()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.dropout / 2)(x)
        
        # Output layer
        if self.output_size == 1:
            outputs = layers.Dense(1, activation='linear')(x)
        else:
            outputs = layers.Dense(self.output_size, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile
        if self.output_size == 1:
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config.ml.dqn_learning_rate),
                loss='mse',
                metrics=['mae']
            )
        else:
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config.ml.dqn_learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        logger.info(f"Built LSTM model with {self.model.count_params()} parameters")
    
    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = None, batch_size: int = None,
              validation_split: float = None) -> Dict:
        """Train the LSTM model"""
        if not TF_AVAILABLE or self.model is None:
            logger.error("TensorFlow not available")
            return {}
        
        epochs = epochs or config.ml.training_epochs
        batch_size = batch_size or config.ml.batch_size
        validation_split = validation_split or config.ml.validation_split
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.ml.early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'loss': self.history.history['loss'][-1],
            'val_loss': self.history.history['val_loss'][-1],
            'epochs_trained': len(self.history.history['loss'])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            return np.array([])
        return self.model.predict(X, verbose=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return self.predict(X)
    
    def save(self, path: str):
        """Save model to disk"""
        if self.model:
            self.model.save(path)
            # Save scaler
            with open(f"{path}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        if TF_AVAILABLE and os.path.exists(path):
            self.model = keras.models.load_model(path)
            scaler_path = f"{path}_scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            logger.info(f"Model loaded from {path}")


# ==================== Transformer Model (PyTorch) ====================

PositionalEncoding = None
TransformerPredictor = None
TransformerTrainer = None
DQNNetwork = None
DQNAgent = None

if TORCH_AVAILABLE:
    class _PositionalEncoding(nn.Module):
        """Positional encoding for Transformer"""
        
        def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)
    
    PositionalEncoding = _PositionalEncoding

    class _TransformerPredictor(nn.Module):
        """Transformer model for price prediction"""
        
        def __init__(self, input_dim: int = 50, d_model: int = None,
                     nhead: int = None, num_layers: int = None,
                     dropout: float = None, output_size: int = 3):
            super().__init__()
            
            self.d_model = d_model or config.ml.transformer_d_model
            self.nhead = nhead or config.ml.transformer_nhead
            self.num_layers = num_layers or config.ml.transformer_num_layers
            self.dropout = dropout or config.ml.transformer_dropout
            self.output_size = output_size
            
            self.input_projection = nn.Linear(input_dim, self.d_model)
            self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.d_model * 4,
                dropout=self.dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            
            self.fc1 = nn.Linear(self.d_model, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc_out = nn.Linear(32, output_size)
            self.dropout_layer = nn.Dropout(self.dropout)
        
        def forward(self, x):
            x = self.input_projection(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x[:, -1, :]
            x = F.relu(self.fc1(x))
            x = self.dropout_layer(x)
            x = F.relu(self.fc2(x))
            x = self.dropout_layer(x)
            x = self.fc_out(x)
            if self.output_size > 1:
                x = F.softmax(x, dim=-1)
            return x
    
    TransformerPredictor = _TransformerPredictor

    class _TransformerTrainer:
        """Trainer for Transformer model"""
        
        def __init__(self, model=None, input_dim: int = 50, output_size: int = 3):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = model or TransformerPredictor(input_dim=input_dim, output_size=output_size)
            self.model.to(self.device)
            
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.ml.dqn_learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
            
            if output_size == 1:
                self.criterion = nn.MSELoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
            
            self.scaler = StandardScaler()
            self.history = {'train_loss': [], 'val_loss': []}
        
        def train(self, X: np.ndarray, y: np.ndarray,
                  epochs: int = None, batch_size: int = None,
                  validation_split: float = None) -> Dict:
            """Train the Transformer model"""
            epochs = epochs or config.ml.training_epochs
            batch_size = batch_size or config.ml.batch_size
            validation_split = validation_split or config.ml.validation_split
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, shuffle=False
            )
            
            X_train = torch.FloatTensor(X_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_train = torch.LongTensor(y_train).to(self.device)
            y_val = torch.LongTensor(y_val).to(self.device)
            
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = self.criterion(val_outputs, y_val).item()
                
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.ml.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            return {'train_loss': train_loss, 'val_loss': val_loss, 'epochs_trained': epoch + 1}
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Make predictions"""
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = self.model(X_tensor)
                return outputs.cpu().numpy()
        
        def save(self, path: str):
            """Save model"""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler': self.scaler,
                'history': self.history
            }, path)
            logger.info(f"Transformer model saved to {path}")
        
        def load(self, path: str):
            """Load model"""
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scaler = checkpoint.get('scaler', StandardScaler())
                self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
                logger.info(f"Transformer model loaded from {path}")
    
    TransformerTrainer = _TransformerTrainer


# ==================== DQN Reinforcement Learning ====================

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = None):
        self.capacity = capacity or config.ml.dqn_memory_size
        self.buffer = deque(maxlen=self.capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List:
        """Sample batch from buffer"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:
    class _DQNNetwork(nn.Module):
        """Deep Q-Network"""
        
        def __init__(self, state_size: int = None, action_size: int = None):
            super().__init__()
            
            self.state_size = state_size or config.ml.dqn_state_size
            self.action_size = action_size or config.ml.dqn_action_size
            
            self.feature = nn.Sequential(
                nn.Linear(self.state_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            )
            
            self.value_stream = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_size)
            )
        
        def forward(self, x):
            features = self.feature(x)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return q_values
    
    DQNNetwork = _DQNNetwork

    class _DQNAgent:
        """DQN Agent for trading decisions"""
        
        def __init__(self, state_size: int = None, action_size: int = None):
            self.state_size = state_size or config.ml.dqn_state_size
            self.action_size = action_size or config.ml.dqn_action_size
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.policy_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
            self.target_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            
            self.optimizer = optim.Adam(
                self.policy_net.parameters(), 
                lr=config.ml.dqn_learning_rate
            )
            
            self.memory = ReplayBuffer()
            
            self.gamma = config.ml.dqn_gamma
            self.epsilon = config.ml.dqn_epsilon_start
            self.epsilon_end = config.ml.dqn_epsilon_end
            self.epsilon_decay = config.ml.dqn_epsilon_decay
            self.batch_size = config.ml.dqn_batch_size
            self.target_update = config.ml.dqn_target_update
            
            self.steps_done = 0
            self.training_history = {'loss': [], 'reward': [], 'epsilon': []}
        
        def select_action(self, state: np.ndarray, training: bool = True) -> int:
            """Select action using epsilon-greedy policy"""
            if training and random.random() < self.epsilon:
                return random.randrange(self.action_size)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        
        def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
            """Get Q-values as action probabilities"""
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                probs = F.softmax(q_values, dim=-1)
                return probs.cpu().numpy()[0]
        
        def store_transition(self, state, action, reward, next_state, done):
            """Store transition in replay buffer"""
            self.memory.push(state, action, reward, next_state, done)
        
        def train_step(self) -> float:
            """Perform one training step"""
            if len(self.memory) < self.batch_size:
                return 0.0
            
            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
            
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
                target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))
            
            loss = F.smooth_l1_loss(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            self.steps_done += 1
            if self.steps_done % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            return loss.item()
        
        def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, float]:
            """Train for one episode"""
            state = env.reset()
            total_reward = 0.0
            total_loss = 0.0
            
            for step in range(max_steps):
                action = self.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                self.store_transition(state, action, reward, next_state, done)
                loss = self.train_step()
                
                total_reward += reward
                total_loss += loss
                state = next_state
                
                if done:
                    break
            
            self.training_history['reward'].append(total_reward)
            self.training_history['loss'].append(total_loss / (step + 1))
            self.training_history['epsilon'].append(self.epsilon)
            
            return total_reward, total_loss / (step + 1)
        
        def save(self, path: str):
            """Save agent"""
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps_done': self.steps_done,
                'history': self.training_history
            }, path)
            logger.info(f"DQN agent saved to {path}")
        
        def load(self, path: str):
            """Load agent"""
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
                self.steps_done = checkpoint.get('steps_done', 0)
                self.training_history = checkpoint.get('history', self.training_history)
                logger.info(f"DQN agent loaded from {path}")
    
    DQNAgent = _DQNAgent


# ==================== Trading Environment for RL ====================

class TradingEnvironment:
    """Trading environment for reinforcement learning"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100.0,
                 max_position: float = 1.0, transaction_cost: float = 0.0001):
        self.data = data
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # -1 to 1 (short to long)
        self.entry_price = 0.0
        self.total_profit = 0.0
        self.trades = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state"""
        if self.current_step >= len(self.data):
            return np.zeros(config.ml.dqn_state_size)
        
        # Get price features
        lookback = min(50, self.current_step + 1)
        price_data = self.data.iloc[self.current_step - lookback + 1:self.current_step + 1]
        
        # Normalize prices
        close_prices = price_data['close'].values
        if len(close_prices) > 0:
            normalized_prices = (close_prices - close_prices.mean()) / (close_prices.std() + 1e-8)
        else:
            normalized_prices = np.zeros(lookback)
        
        # Pad if necessary
        if len(normalized_prices) < 50:
            normalized_prices = np.pad(normalized_prices, (50 - len(normalized_prices), 0))
        
        # Add position and balance info
        state = np.concatenate([
            normalized_prices[-45:],  # Last 45 price points
            [self.position],  # Current position
            [self.balance / self.initial_balance - 1],  # Normalized balance
            [self.total_profit / self.initial_balance],  # Normalized profit
            [self.entry_price / self.data['close'].iloc[self.current_step] - 1 if self.entry_price > 0 else 0],  # Unrealized PnL
            [len(self.trades) / 100]  # Trade count normalized
        ])
        
        return state[:config.ml.dqn_state_size]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return new state, reward, done, info"""
        # Actions: 0 = Sell/Short, 1 = Hold, 2 = Buy/Long
        
        current_price = self.data['close'].iloc[self.current_step]
        reward = 0.0
        
        # Execute action
        if action == 0 and self.position >= 0:  # Sell/Short
            if self.position > 0:
                # Close long position
                profit = (current_price - self.entry_price) * self.position * self.balance
                profit -= self.transaction_cost * self.balance
                self.balance += profit
                self.total_profit += profit
                reward = profit / self.initial_balance
                self.trades.append({
                    'type': 'close_long',
                    'price': current_price,
                    'profit': profit
                })
            # Open short
            self.position = -self.max_position
            self.entry_price = current_price
            self.balance -= self.transaction_cost * self.balance
            
        elif action == 2 and self.position <= 0:  # Buy/Long
            if self.position < 0:
                # Close short position
                profit = (self.entry_price - current_price) * abs(self.position) * self.balance
                profit -= self.transaction_cost * self.balance
                self.balance += profit
                self.total_profit += profit
                reward = profit / self.initial_balance
                self.trades.append({
                    'type': 'close_short',
                    'price': current_price,
                    'profit': profit
                })
            # Open long
            self.position = self.max_position
            self.entry_price = current_price
            self.balance -= self.transaction_cost * self.balance
        
        # Move to next step
        self.current_step += 1
        
        # Calculate unrealized PnL for reward shaping
        if self.position != 0 and self.current_step < len(self.data):
            next_price = self.data['close'].iloc[self.current_step]
            if self.position > 0:
                unrealized = (next_price - current_price) / current_price
            else:
                unrealized = (current_price - next_price) / current_price
            reward += unrealized * 0.1  # Small reward for unrealized gains
        
        # Check if done
        done = self.current_step >= len(self.data) - 1 or self.balance <= 0
        
        # Info
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_profit': self.total_profit,
            'num_trades': len(self.trades)
        }
        
        return self._get_state(), reward, done, info


# ==================== Ensemble Model ====================

class EnsemblePredictor:
    """Ensemble of multiple models for robust predictions"""
    
    def __init__(self, input_shape: Tuple[int, int] = None):
        self.input_shape = input_shape or (config.ml.lstm_sequence_length, 50)
        
        self.lstm_model = None
        self.transformer_trainer = None
        self.dqn_agent = None
        
        self.weights = {'lstm': 0.4, 'transformer': 0.3, 'dqn': 0.3}
        self.initialized = False
    
    def initialize_models(self, input_dim: int = 50):
        """Initialize all models"""
        if TF_AVAILABLE:
            self.lstm_model = LSTMPricePredictor(
                input_shape=self.input_shape,
                output_size=3
            )
        
        if TORCH_AVAILABLE:
            self.transformer_trainer = TransformerTrainer(
                input_dim=input_dim,
                output_size=3
            )
            self.dqn_agent = DQNAgent(
                state_size=config.ml.dqn_state_size,
                action_size=3
            )
        
        self.initialized = True
        logger.info("Ensemble models initialized")
    
    def train_all(self, X: np.ndarray, y: np.ndarray, 
                  env_data: pd.DataFrame = None) -> Dict:
        """Train all models"""
        results = {}
        
        # Train LSTM
        if self.lstm_model:
            logger.info("Training LSTM model...")
            results['lstm'] = self.lstm_model.train(X, y)
        
        # Train Transformer
        if self.transformer_trainer:
            logger.info("Training Transformer model...")
            results['transformer'] = self.transformer_trainer.train(X, y)
        
        # Train DQN
        if self.dqn_agent and env_data is not None:
            logger.info("Training DQN agent...")
            env = TradingEnvironment(env_data)
            total_reward = 0
            for episode in range(50):  # 50 episodes
                reward, loss = self.dqn_agent.train_episode(env)
                total_reward += reward
                if (episode + 1) % 10 == 0:
                    logger.info(f"DQN Episode {episode + 1}/50 - Reward: {reward:.4f}")
            results['dqn'] = {'total_reward': total_reward / 50}
        
        return results
    
    def predict(self, X: np.ndarray, state: np.ndarray = None) -> Tuple[int, float]:
        """Get ensemble prediction"""
        predictions = []
        confidences = []
        
        # LSTM prediction
        if self.lstm_model:
            lstm_pred = self.lstm_model.predict(X)
            if len(lstm_pred) > 0:
                predictions.append(('lstm', lstm_pred[-1]))
                confidences.append(np.max(lstm_pred[-1]))
        
        # Transformer prediction
        if self.transformer_trainer:
            trans_pred = self.transformer_trainer.predict(X)
            if len(trans_pred) > 0:
                predictions.append(('transformer', trans_pred[-1]))
                confidences.append(np.max(trans_pred[-1]))
        
        # DQN prediction
        if self.dqn_agent and state is not None:
            dqn_probs = self.dqn_agent.get_action_probabilities(state)
            predictions.append(('dqn', dqn_probs))
            confidences.append(np.max(dqn_probs))
        
        if not predictions:
            return 1, 0.0  # Default to Hold
        
        # Weighted ensemble
        weighted_probs = np.zeros(3)
        for name, probs in predictions:
            weight = self.weights.get(name, 0.33)
            weighted_probs += weight * np.array(probs)
        
        weighted_probs /= sum(self.weights.values())
        
        action = np.argmax(weighted_probs)
        confidence = weighted_probs[action]
        
        return int(action), float(confidence)
    
    def save_all(self, directory: str):
        """Save all models"""
        os.makedirs(directory, exist_ok=True)
        
        if self.lstm_model:
            self.lstm_model.save(os.path.join(directory, 'lstm_model'))
        
        if self.transformer_trainer:
            self.transformer_trainer.save(os.path.join(directory, 'transformer_model.pt'))
        
        if self.dqn_agent:
            self.dqn_agent.save(os.path.join(directory, 'dqn_agent.pt'))
        
        # Save weights
        with open(os.path.join(directory, 'ensemble_weights.json'), 'w') as f:
            json.dump(self.weights, f)
        
        logger.info(f"All models saved to {directory}")
    
    def load_all(self, directory: str):
        """Load all models"""
        if self.lstm_model and os.path.exists(os.path.join(directory, 'lstm_model')):
            self.lstm_model.load(os.path.join(directory, 'lstm_model'))
        
        if self.transformer_trainer and os.path.exists(os.path.join(directory, 'transformer_model.pt')):
            self.transformer_trainer.load(os.path.join(directory, 'transformer_model.pt'))
        
        if self.dqn_agent and os.path.exists(os.path.join(directory, 'dqn_agent.pt')):
            self.dqn_agent.load(os.path.join(directory, 'dqn_agent.pt'))
        
        weights_path = os.path.join(directory, 'ensemble_weights.json')
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                self.weights = json.load(f)
        
        logger.info(f"All models loaded from {directory}")


# Singleton instance
ensemble_predictor = EnsemblePredictor()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ML models...")
    print(f"TensorFlow available: {TF_AVAILABLE}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    
    # Generate dummy data
    np.random.seed(42)
    X = np.random.randn(1000, 60, 50)
    y = np.random.randint(0, 3, 1000)
    
    # Test LSTM
    if TF_AVAILABLE:
        print("\nTesting LSTM...")
        lstm = LSTMPricePredictor(input_shape=(60, 50), output_size=3)
        result = lstm.train(X[:100], y[:100], epochs=2)
        print(f"LSTM training result: {result}")
    
    # Test Transformer
    if TORCH_AVAILABLE:
        print("\nTesting Transformer...")
        transformer = TransformerTrainer(input_dim=50, output_size=3)
        result = transformer.train(X[:100], y[:100], epochs=2)
        print(f"Transformer training result: {result}")
    
    # Test DQN
    if TORCH_AVAILABLE:
        print("\nTesting DQN...")
        # Create dummy price data
        price_data = pd.DataFrame({
            'open': np.random.randn(500).cumsum() + 100,
            'high': np.random.randn(500).cumsum() + 101,
            'low': np.random.randn(500).cumsum() + 99,
            'close': np.random.randn(500).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 500)
        })
        
        env = TradingEnvironment(price_data)
        agent = DQNAgent()
        
        for ep in range(3):
            reward, loss = agent.train_episode(env, max_steps=100)
            print(f"Episode {ep + 1}: Reward={reward:.4f}, Loss={loss:.4f}")
    
    print("\nML models test complete!")
