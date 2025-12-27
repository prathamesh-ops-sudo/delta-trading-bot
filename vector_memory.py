"""
Vector Memory Module - FAISS-based Long-Term Episodic Memory
Phoenix Trading System - Experience Replay and Insight Retrieval

This module provides:
- FAISS vector database for storing trade experiences and insights
- Semantic search for retrieving relevant past experiences
- Transformer-based summarization of trading sessions
- Experience replay buffer for RL training
"""

import logging
import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available - using simple similarity search fallback")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available - using TF-IDF fallback")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - using keyword matching fallback")


@dataclass
class TradeExperience:
    """A single trade experience for memory storage"""
    experience_id: str
    timestamp: datetime
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    exit_price: float
    profit_loss: float
    profit_pips: float
    strategy: str
    regime: str
    confidence: float
    indicators: Dict[str, float]
    market_context: str  # Natural language description
    outcome_analysis: str  # What happened and why
    lessons_learned: str  # Key takeaways
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['embedding'] = None  # Don't serialize embedding
        return d
    
    def to_searchable_text(self) -> str:
        """Convert experience to searchable text"""
        return f"""
        Trade: {self.symbol} {self.direction} using {self.strategy} strategy
        Regime: {self.regime}, Confidence: {self.confidence:.0%}
        Result: {'Profit' if self.profit_loss > 0 else 'Loss'} of {abs(self.profit_pips):.1f} pips
        Context: {self.market_context}
        Analysis: {self.outcome_analysis}
        Lessons: {self.lessons_learned}
        RSI: {self.indicators.get('rsi', 'N/A')}, ADX: {self.indicators.get('adx', 'N/A')}
        """


@dataclass
class TradingSession:
    """Summary of a trading session"""
    session_id: str
    date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_pips: float
    best_trade: Optional[str]
    worst_trade: Optional[str]
    dominant_strategy: str
    dominant_regime: str
    session_summary: str  # Natural language summary
    key_insights: List[str]
    mood: str  # confident, cautious, frustrated, learning
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['date'] = self.date.isoformat()
        d['embedding'] = None
        return d
    
    def to_searchable_text(self) -> str:
        """Convert session to searchable text"""
        return f"""
        Trading Session: {self.date.strftime('%Y-%m-%d')}
        Results: {self.winning_trades}W/{self.losing_trades}L, {self.total_pnl:+.2f} USD, {self.total_pips:+.1f} pips
        Strategy: {self.dominant_strategy}, Regime: {self.dominant_regime}
        Mood: {self.mood}
        Summary: {self.session_summary}
        Insights: {'; '.join(self.key_insights)}
        """


class EmbeddingEngine:
    """Generates embeddings for text using available methods"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.tfidf = None
        self.dimension = 384  # Default for MiniLM
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded SentenceTransformer: {model_name} (dim={self.dimension})")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer: {e}")
        
        if self.model is None and SKLEARN_AVAILABLE:
            self.tfidf = TfidfVectorizer(max_features=self.dimension)
            self.tfidf_fitted = False
            logger.info("Using TF-IDF fallback for embeddings")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        if not texts:
            return np.array([])
        
        if self.model:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        
        if self.tfidf:
            if not self.tfidf_fitted:
                # Fit on first batch
                self.tfidf.fit(texts)
                self.tfidf_fitted = True
            
            try:
                embeddings = self.tfidf.transform(texts).toarray()
                # Pad or truncate to target dimension
                if embeddings.shape[1] < self.dimension:
                    padding = np.zeros((embeddings.shape[0], self.dimension - embeddings.shape[1]))
                    embeddings = np.hstack([embeddings, padding])
                elif embeddings.shape[1] > self.dimension:
                    embeddings = embeddings[:, :self.dimension]
                return embeddings
            except:
                pass
        
        # Fallback: random embeddings (not useful but prevents crashes)
        logger.warning("Using random embeddings as fallback")
        return np.random.randn(len(texts), self.dimension).astype(np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        return self.encode([text])[0]


class VectorMemory:
    """
    FAISS-based vector memory for storing and retrieving trade experiences.
    Falls back to simple cosine similarity if FAISS is not available.
    """
    
    def __init__(self, storage_path: str = "./data/vector_memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding engine
        self.embedding_engine = EmbeddingEngine()
        self.dimension = self.embedding_engine.dimension
        
        # Initialize FAISS index or fallback
        self.index = None
        self.use_faiss = FAISS_AVAILABLE
        
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine sim with normalized vectors)
            logger.info(f"FAISS index initialized (dim={self.dimension})")
        
        # Storage for experiences and sessions
        self.experiences: List[TradeExperience] = []
        self.sessions: List[TradingSession] = []
        self.experience_embeddings: List[np.ndarray] = []
        self.session_embeddings: List[np.ndarray] = []
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load existing experiences and sessions from disk"""
        experiences_file = self.storage_path / "experiences.json"
        sessions_file = self.storage_path / "sessions.json"
        
        try:
            if experiences_file.exists():
                with open(experiences_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        try:
                            exp = TradeExperience(
                                experience_id=item['experience_id'],
                                timestamp=datetime.fromisoformat(item['timestamp']),
                                symbol=item['symbol'],
                                direction=item['direction'],
                                entry_price=item['entry_price'],
                                exit_price=item['exit_price'],
                                profit_loss=item['profit_loss'],
                                profit_pips=item['profit_pips'],
                                strategy=item['strategy'],
                                regime=item['regime'],
                                confidence=item['confidence'],
                                indicators=item['indicators'],
                                market_context=item['market_context'],
                                outcome_analysis=item['outcome_analysis'],
                                lessons_learned=item['lessons_learned']
                            )
                            self.experiences.append(exp)
                        except Exception as e:
                            logger.warning(f"Could not load experience: {e}")
                
                # Rebuild embeddings
                if self.experiences:
                    texts = [exp.to_searchable_text() for exp in self.experiences]
                    self.experience_embeddings = list(self.embedding_engine.encode(texts))
                    
                    if self.use_faiss and self.experience_embeddings:
                        embeddings_array = np.array(self.experience_embeddings).astype(np.float32)
                        faiss.normalize_L2(embeddings_array)
                        self.index.add(embeddings_array)
                
                logger.info(f"Loaded {len(self.experiences)} experiences from disk")
        except Exception as e:
            logger.warning(f"Could not load experiences: {e}")
        
        try:
            if sessions_file.exists():
                with open(sessions_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        try:
                            session = TradingSession(
                                session_id=item['session_id'],
                                date=datetime.fromisoformat(item['date']),
                                total_trades=item['total_trades'],
                                winning_trades=item['winning_trades'],
                                losing_trades=item['losing_trades'],
                                total_pnl=item['total_pnl'],
                                total_pips=item['total_pips'],
                                best_trade=item.get('best_trade'),
                                worst_trade=item.get('worst_trade'),
                                dominant_strategy=item['dominant_strategy'],
                                dominant_regime=item['dominant_regime'],
                                session_summary=item['session_summary'],
                                key_insights=item['key_insights'],
                                mood=item['mood']
                            )
                            self.sessions.append(session)
                        except Exception as e:
                            logger.warning(f"Could not load session: {e}")
                
                # Rebuild session embeddings
                if self.sessions:
                    texts = [s.to_searchable_text() for s in self.sessions]
                    self.session_embeddings = list(self.embedding_engine.encode(texts))
                
                logger.info(f"Loaded {len(self.sessions)} sessions from disk")
        except Exception as e:
            logger.warning(f"Could not load sessions: {e}")
    
    def _save_data(self):
        """Save experiences and sessions to disk"""
        experiences_file = self.storage_path / "experiences.json"
        sessions_file = self.storage_path / "sessions.json"
        
        try:
            with open(experiences_file, 'w') as f:
                json.dump([exp.to_dict() for exp in self.experiences], f, indent=2)
            
            with open(sessions_file, 'w') as f:
                json.dump([s.to_dict() for s in self.sessions], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving vector memory: {e}")
    
    def add_experience(self, experience: TradeExperience):
        """Add a trade experience to memory"""
        # Generate embedding
        text = experience.to_searchable_text()
        embedding = self.embedding_engine.encode_single(text)
        
        # Add to storage
        self.experiences.append(experience)
        self.experience_embeddings.append(embedding)
        
        # Add to FAISS index
        if self.use_faiss:
            embedding_normalized = embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(embedding_normalized)
            self.index.add(embedding_normalized)
        
        # Save to disk
        self._save_data()
        
        logger.debug(f"Added experience {experience.experience_id} to vector memory")
    
    def add_session(self, session: TradingSession):
        """Add a trading session to memory"""
        # Generate embedding
        text = session.to_searchable_text()
        embedding = self.embedding_engine.encode_single(text)
        
        # Add to storage
        self.sessions.append(session)
        self.session_embeddings.append(embedding)
        
        # Save to disk
        self._save_data()
        
        logger.debug(f"Added session {session.session_id} to vector memory")
    
    def search_similar_experiences(
        self, 
        query: str, 
        k: int = 5,
        symbol_filter: Optional[str] = None,
        strategy_filter: Optional[str] = None
    ) -> List[Tuple[TradeExperience, float]]:
        """
        Search for similar trade experiences.
        Returns list of (experience, similarity_score) tuples.
        """
        if not self.experiences:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_engine.encode_single(query)
        
        if self.use_faiss and self.index.ntotal > 0:
            # FAISS search
            query_normalized = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_normalized)
            
            # Search more than k to allow for filtering
            search_k = min(k * 3, len(self.experiences))
            distances, indices = self.index.search(query_normalized, search_k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.experiences):
                    continue
                
                exp = self.experiences[idx]
                
                # Apply filters
                if symbol_filter and exp.symbol != symbol_filter:
                    continue
                if strategy_filter and exp.strategy != strategy_filter:
                    continue
                
                results.append((exp, float(dist)))
                
                if len(results) >= k:
                    break
            
            return results
        
        else:
            # Fallback: cosine similarity
            results = []
            
            for i, exp in enumerate(self.experiences):
                # Apply filters
                if symbol_filter and exp.symbol != symbol_filter:
                    continue
                if strategy_filter and exp.strategy != strategy_filter:
                    continue
                
                # Calculate similarity
                if i < len(self.experience_embeddings):
                    exp_embedding = self.experience_embeddings[i]
                    similarity = np.dot(query_embedding, exp_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(exp_embedding) + 1e-8
                    )
                    results.append((exp, float(similarity)))
            
            # Sort by similarity and return top k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
    
    def search_similar_sessions(self, query: str, k: int = 3) -> List[Tuple[TradingSession, float]]:
        """Search for similar trading sessions"""
        if not self.sessions:
            return []
        
        query_embedding = self.embedding_engine.encode_single(query)
        
        results = []
        for i, session in enumerate(self.sessions):
            if i < len(self.session_embeddings):
                session_embedding = self.session_embeddings[i]
                similarity = np.dot(query_embedding, session_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(session_embedding) + 1e-8
                )
                results.append((session, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def get_relevant_wisdom(
        self, 
        symbol: str, 
        strategy: str, 
        regime: str,
        indicators: Dict[str, float]
    ) -> str:
        """
        Get relevant wisdom from past experiences for a potential trade.
        Returns natural language advice based on similar past situations.
        """
        # Build query from current situation
        query = f"""
        Trading {symbol} with {strategy} strategy in {regime} regime.
        RSI: {indicators.get('rsi', 'unknown')}, ADX: {indicators.get('adx', 'unknown')}
        Looking for similar past trades and their outcomes.
        """
        
        # Search for similar experiences
        similar = self.search_similar_experiences(query, k=5, symbol_filter=symbol)
        
        if not similar:
            return "No similar past experiences found. Proceed with standard analysis."
        
        # Analyze past outcomes
        wins = [exp for exp, _ in similar if exp.profit_loss > 0]
        losses = [exp for exp, _ in similar if exp.profit_loss <= 0]
        
        win_rate = len(wins) / len(similar) if similar else 0
        
        # Build wisdom response
        wisdom_parts = []
        
        if win_rate >= 0.7:
            wisdom_parts.append(f"Strong historical performance ({win_rate:.0%} win rate) in similar situations.")
        elif win_rate <= 0.3:
            wisdom_parts.append(f"Caution: Poor historical performance ({win_rate:.0%} win rate) in similar situations.")
        else:
            wisdom_parts.append(f"Mixed historical results ({win_rate:.0%} win rate) in similar situations.")
        
        # Extract key lessons from similar trades
        lessons = []
        for exp, score in similar[:3]:
            if exp.lessons_learned and len(exp.lessons_learned) > 10:
                lessons.append(exp.lessons_learned)
        
        if lessons:
            wisdom_parts.append("Key lessons from similar trades:")
            for lesson in lessons[:2]:
                wisdom_parts.append(f"  - {lesson[:100]}")
        
        return "\n".join(wisdom_parts)
    
    def get_experience_replay_batch(self, batch_size: int = 32) -> List[TradeExperience]:
        """
        Get a batch of experiences for RL training.
        Prioritizes recent experiences and diverse outcomes.
        """
        if not self.experiences:
            return []
        
        # Weight recent experiences more heavily
        weights = []
        now = datetime.utcnow()
        
        for exp in self.experiences:
            age_days = (now - exp.timestamp).days
            recency_weight = 1.0 / (1.0 + age_days * 0.1)  # Decay over time
            
            # Also weight by outcome diversity
            outcome_weight = 1.0 if exp.profit_loss != 0 else 0.5
            
            weights.append(recency_weight * outcome_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(self.experiences)] * len(self.experiences)
        
        # Sample with replacement
        indices = np.random.choice(
            len(self.experiences),
            size=min(batch_size, len(self.experiences)),
            replace=False,
            p=weights
        )
        
        return [self.experiences[i] for i in indices]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored experiences"""
        if not self.experiences:
            return {
                'total_experiences': 0,
                'total_sessions': len(self.sessions),
                'win_rate': 0,
                'avg_pips': 0
            }
        
        wins = sum(1 for exp in self.experiences if exp.profit_loss > 0)
        total_pips = sum(exp.profit_pips for exp in self.experiences)
        
        return {
            'total_experiences': len(self.experiences),
            'total_sessions': len(self.sessions),
            'win_rate': wins / len(self.experiences),
            'avg_pips': total_pips / len(self.experiences),
            'symbols_traded': list(set(exp.symbol for exp in self.experiences)),
            'strategies_used': list(set(exp.strategy for exp in self.experiences)),
            'oldest_experience': min(exp.timestamp for exp in self.experiences).isoformat(),
            'newest_experience': max(exp.timestamp for exp in self.experiences).isoformat()
        }


class ExperienceReplayBuffer:
    """
    Experience replay buffer for RL training.
    Stores state-action-reward-next_state tuples.
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Dict] = []
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict] = None
    ):
        """Add experience to buffer"""
        experience = {
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'done': done,
            'info': info or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample a batch of experiences"""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def prioritized_sample(self, batch_size: int, alpha: float = 0.6) -> List[Dict]:
        """
        Prioritized experience replay sampling.
        Higher reward magnitude = higher priority.
        """
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        
        # Calculate priorities based on reward magnitude
        priorities = np.array([abs(exp['reward']) + 0.01 for exp in self.buffer])
        priorities = priorities ** alpha
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probabilities)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath: str):
        """Save buffer to file"""
        with open(filepath, 'w') as f:
            json.dump(self.buffer, f)
    
    def load(self, filepath: str):
        """Load buffer from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.buffer = json.load(f)
            self.position = len(self.buffer) % self.capacity


# Global instance
vector_memory = VectorMemory()
replay_buffer = ExperienceReplayBuffer()


if __name__ == "__main__":
    # Test the vector memory
    logging.basicConfig(level=logging.INFO)
    
    memory = VectorMemory()
    
    # Add a test experience
    test_exp = TradeExperience(
        experience_id="test_001",
        timestamp=datetime.utcnow(),
        symbol="EURUSD",
        direction="LONG",
        entry_price=1.0850,
        exit_price=1.0880,
        profit_loss=30.0,
        profit_pips=30.0,
        strategy="momentum",
        regime="trending",
        confidence=0.75,
        indicators={'rsi': 45, 'adx': 32, 'atr': 0.0015},
        market_context="Strong uptrend with bullish momentum, RSI not overbought",
        outcome_analysis="Trade worked well as trend continued. Entry was good timing after pullback.",
        lessons_learned="Momentum trades in strong trends work best when RSI is between 40-60"
    )
    
    memory.add_experience(test_exp)
    
    # Search for similar
    results = memory.search_similar_experiences("EURUSD momentum trade in trending market")
    print(f"Found {len(results)} similar experiences")
    
    # Get wisdom
    wisdom = memory.get_relevant_wisdom("EURUSD", "momentum", "trending", {'rsi': 50, 'adx': 30})
    print(f"Wisdom: {wisdom}")
    
    # Get stats
    stats = memory.get_statistics()
    print(f"Stats: {stats}")
