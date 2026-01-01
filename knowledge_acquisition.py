"""
Knowledge Acquisition Module - 24/7 Off-Market Learning
Phoenix Trading System - Continuous Evolution Engine

This module scrapes and analyzes:
- RSS feeds from major financial news sources
- Reddit (r/Forex, r/algotrading, r/Daytrading)
- NewsAPI for real-time sentiment
- Forex Factory calendar for high-impact events

All knowledge is distilled into actionable insights that feed into
the RL brain for continuous improvement even when markets are closed.
"""

import logging
import json
import os
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import time
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.warning("feedparser not available - RSS feeds disabled")

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logger.warning("praw not available - Reddit integration disabled")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available - API calls disabled")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available - using VADER fallback")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not available - using basic sentiment")


@dataclass
class MarketInsight:
    """A single piece of market knowledge/insight"""
    source: str  # rss, reddit, news, calendar
    title: str
    content: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # bearish, neutral, bullish
    relevance_score: float  # 0 to 1
    symbols_mentioned: List[str]
    timestamp: datetime
    url: Optional[str] = None
    impact_level: str = "low"  # low, medium, high
    insight_type: str = "general"  # news, analysis, sentiment, event
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class KnowledgeState:
    """Current state of market knowledge"""
    overall_sentiment: float = 0.0
    sentiment_by_symbol: Dict[str, float] = field(default_factory=dict)
    upcoming_events: List[Dict] = field(default_factory=list)
    trending_topics: List[str] = field(default_factory=list)
    strategy_hints: List[str] = field(default_factory=list)
    last_update: Optional[datetime] = None
    insights_count: int = 0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['last_update'] = self.last_update.isoformat() if self.last_update else None
        return d


class SentimentAnalyzer:
    """Multi-method sentiment analysis"""
    
    def __init__(self):
        self.vader = None
        self.finbert = None
        
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        
        # FinBERT is heavy - only load if transformers available and explicitly requested
        self.finbert_available = TRANSFORMERS_AVAILABLE
    
    def analyze(self, text: str) -> Tuple[float, str]:
        """
        Analyze sentiment of text.
        Returns (score, label) where score is -1 to 1 and label is bearish/neutral/bullish
        """
        if not text or len(text.strip()) < 10:
            return 0.0, "neutral"
        
        # Use VADER if available (fast and good for social media)
        if self.vader:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                label = "bullish"
            elif compound <= -0.05:
                label = "bearish"
            else:
                label = "neutral"
            
            return compound, label
        
        # Fallback to keyword-based sentiment
        return self._keyword_sentiment(text)
    
    def _keyword_sentiment(self, text: str) -> Tuple[float, str]:
        """Simple keyword-based sentiment as fallback"""
        text_lower = text.lower()
        
        bullish_words = [
            'bullish', 'buy', 'long', 'rally', 'surge', 'breakout', 'support',
            'uptrend', 'higher', 'gains', 'profit', 'strong', 'momentum',
            'recovery', 'growth', 'optimistic', 'positive'
        ]
        
        bearish_words = [
            'bearish', 'sell', 'short', 'crash', 'drop', 'breakdown', 'resistance',
            'downtrend', 'lower', 'losses', 'weak', 'decline', 'fall',
            'recession', 'pessimistic', 'negative', 'risk-off'
        ]
        
        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, "neutral"
        
        score = (bullish_count - bearish_count) / total
        
        if score > 0.2:
            label = "bullish"
        elif score < -0.2:
            label = "bearish"
        else:
            label = "neutral"
        
        return score, label


class RSSFeedCollector:
    """Collects and processes RSS feeds from financial news sources"""
    
    # Reliable RSS feeds for Forex/financial news
    DEFAULT_FEEDS = {
        'forex_factory': 'https://www.forexfactory.com/ffcal_week_this.xml',
        'investing_forex': 'https://www.investing.com/rss/news_14.rss',
        'fxstreet': 'https://www.fxstreet.com/rss/news',
        'dailyfx': 'https://www.dailyfx.com/feeds/market-news',
        'reuters_forex': 'https://www.reuters.com/arc/outboundfeeds/v3/all/?outputType=xml&size=10',
        'bbc_business': 'https://feeds.bbci.co.uk/news/business/rss.xml',
        'cnbc': 'https://www.cnbc.com/id/100727362/device/rss/rss.html',
    }
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        self.sentiment = sentiment_analyzer
        self.seen_articles: set = set()
        self.feeds = self.DEFAULT_FEEDS.copy()
    
    def collect(self, max_articles: int = 50) -> List[MarketInsight]:
        """Collect articles from all RSS feeds"""
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not available, skipping RSS collection")
            return []
        
        insights = []
        
        for feed_name, feed_url in self.feeds.items():
            try:
                articles = self._parse_feed(feed_name, feed_url, max_articles // len(self.feeds))
                insights.extend(articles)
            except Exception as e:
                logger.warning(f"Error parsing feed {feed_name}: {e}")
        
        return insights
    
    def _parse_feed(self, feed_name: str, feed_url: str, max_items: int) -> List[MarketInsight]:
        """Parse a single RSS feed"""
        insights = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:max_items]:
                # Create unique ID to avoid duplicates
                article_id = hashlib.md5(
                    (entry.get('title', '') + entry.get('link', '')).encode()
                ).hexdigest()
                
                if article_id in self.seen_articles:
                    continue
                
                self.seen_articles.add(article_id)
                
                # Extract content
                title = entry.get('title', '')
                summary = entry.get('summary', entry.get('description', ''))
                content = f"{title}. {summary}"
                
                # Analyze sentiment
                score, label = self.sentiment.analyze(content)
                
                # Extract mentioned symbols
                symbols = self._extract_symbols(content)
                
                # Calculate relevance
                relevance = self._calculate_relevance(content, symbols)
                
                # Parse timestamp
                timestamp = self._parse_timestamp(entry)
                
                insight = MarketInsight(
                    source=f"rss_{feed_name}",
                    title=title,
                    content=summary[:500],  # Truncate long content
                    sentiment_score=score,
                    sentiment_label=label,
                    relevance_score=relevance,
                    symbols_mentioned=symbols,
                    timestamp=timestamp,
                    url=entry.get('link'),
                    insight_type="news"
                )
                
                insights.append(insight)
                
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url}: {e}")
        
        return insights
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract Forex symbols mentioned in text"""
        symbols = []
        text_upper = text.upper()
        
        forex_pairs = [
            'EURUSD', 'EUR/USD', 'GBPUSD', 'GBP/USD', 'USDJPY', 'USD/JPY',
            'USDCHF', 'USD/CHF', 'AUDUSD', 'AUD/USD', 'USDCAD', 'USD/CAD',
            'NZDUSD', 'NZD/USD', 'EURGBP', 'EUR/GBP', 'EURJPY', 'EUR/JPY'
        ]
        
        currencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
        
        for pair in forex_pairs:
            if pair in text_upper or pair.replace('/', '') in text_upper:
                normalized = pair.replace('/', '')
                if normalized not in symbols:
                    symbols.append(normalized)
        
        # Also check for currency mentions
        for currency in currencies:
            if currency in text_upper and not any(currency in s for s in symbols):
                # Map single currency to common pairs
                if currency == 'EUR' and 'EURUSD' not in symbols:
                    symbols.append('EURUSD')
                elif currency == 'GBP' and 'GBPUSD' not in symbols:
                    symbols.append('GBPUSD')
                elif currency == 'JPY' and 'USDJPY' not in symbols:
                    symbols.append('USDJPY')
        
        return symbols
    
    def _calculate_relevance(self, text: str, symbols: List[str]) -> float:
        """Calculate relevance score for Forex trading"""
        relevance = 0.3  # Base relevance
        
        # Boost for mentioned symbols
        relevance += min(0.3, len(symbols) * 0.1)
        
        # Boost for trading-related keywords
        trading_keywords = [
            'forex', 'currency', 'exchange rate', 'central bank', 'fed',
            'ecb', 'boe', 'interest rate', 'inflation', 'gdp', 'employment',
            'nfp', 'fomc', 'monetary policy', 'trade balance'
        ]
        
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in trading_keywords if kw in text_lower)
        relevance += min(0.4, keyword_matches * 0.1)
        
        return min(1.0, relevance)
    
    def _parse_timestamp(self, entry: Dict) -> datetime:
        """Parse timestamp from RSS entry"""
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                return datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                return datetime(*entry.updated_parsed[:6])
        except:
            pass
        return datetime.utcnow()


class RedditCollector:
    """Collects insights from Reddit trading communities"""
    
    SUBREDDITS = ['Forex', 'algotrading', 'Daytrading', 'wallstreetbets']
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        self.sentiment = sentiment_analyzer
        self.reddit = None
        self.seen_posts: set = set()
        
        # Initialize Reddit client if credentials available
        if PRAW_AVAILABLE:
            client_id = os.environ.get('REDDIT_CLIENT_ID')
            client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
            
            if client_id and client_secret:
                try:
                    self.reddit = praw.Reddit(
                        client_id=client_id,
                        client_secret=client_secret,
                        user_agent='PhoenixTrader/1.0 (by /u/trading_bot)'
                    )
                    logger.info("Reddit client initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Reddit client: {e}")
    
    def collect(self, max_posts: int = 30) -> List[MarketInsight]:
        """Collect posts from trading subreddits"""
        if not self.reddit:
            logger.debug("Reddit client not available, skipping collection")
            return []
        
        insights = []
        posts_per_sub = max_posts // len(self.SUBREDDITS)
        
        for subreddit_name in self.SUBREDDITS:
            try:
                posts = self._collect_from_subreddit(subreddit_name, posts_per_sub)
                insights.extend(posts)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Error collecting from r/{subreddit_name}: {e}")
        
        return insights
    
    def _collect_from_subreddit(self, subreddit_name: str, limit: int) -> List[MarketInsight]:
        """Collect posts from a single subreddit"""
        insights = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get hot posts
            for post in subreddit.hot(limit=limit):
                if post.id in self.seen_posts:
                    continue
                
                self.seen_posts.add(post.id)
                
                # Combine title and selftext
                content = f"{post.title}. {post.selftext[:500] if post.selftext else ''}"
                
                # Analyze sentiment
                score, label = self.sentiment.analyze(content)
                
                # Extract symbols
                symbols = self._extract_symbols(content)
                
                # Calculate relevance based on engagement
                relevance = self._calculate_relevance(post, symbols)
                
                insight = MarketInsight(
                    source=f"reddit_r/{subreddit_name}",
                    title=post.title,
                    content=content[:500],
                    sentiment_score=score,
                    sentiment_label=label,
                    relevance_score=relevance,
                    symbols_mentioned=symbols,
                    timestamp=datetime.fromtimestamp(post.created_utc),
                    url=f"https://reddit.com{post.permalink}",
                    insight_type="sentiment"
                )
                
                insights.append(insight)
                
        except Exception as e:
            logger.error(f"Error parsing subreddit {subreddit_name}: {e}")
        
        return insights
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract Forex symbols from text"""
        symbols = []
        text_upper = text.upper()
        
        forex_pairs = [
            'EURUSD', 'EUR/USD', 'GBPUSD', 'GBP/USD', 'USDJPY', 'USD/JPY',
            'USDCHF', 'USD/CHF', 'AUDUSD', 'AUD/USD', 'USDCAD', 'USD/CAD'
        ]
        
        for pair in forex_pairs:
            if pair in text_upper or pair.replace('/', '') in text_upper:
                normalized = pair.replace('/', '')
                if normalized not in symbols:
                    symbols.append(normalized)
        
        return symbols
    
    def _calculate_relevance(self, post, symbols: List[str]) -> float:
        """Calculate relevance based on engagement and content"""
        relevance = 0.2
        
        # Boost for symbols mentioned
        relevance += min(0.3, len(symbols) * 0.15)
        
        # Boost for engagement (upvotes)
        if post.score > 100:
            relevance += 0.2
        elif post.score > 50:
            relevance += 0.15
        elif post.score > 20:
            relevance += 0.1
        
        # Boost for comments (discussion)
        if post.num_comments > 50:
            relevance += 0.15
        elif post.num_comments > 20:
            relevance += 0.1
        
        return min(1.0, relevance)


class NewsAPICollector:
    """
    Collects news from NewsAPI.ai (Event Registry)
    
    IMPORTANT: Free tier has only 2000 tokens TOTAL (not monthly).
    We use tokens very conservatively:
    - Only query once per 6 hours (4 queries/day max)
    - Single focused query for forex/currency news
    - Limit to 5 articles per query
    - Each query uses ~1 token, so ~4 tokens/day = ~500 days of usage
    """
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        self.sentiment = sentiment_analyzer
        self.api_key = os.environ.get('NEWSAPI_AI_KEY')  # NewsAPI.ai key
        self.base_url = "https://eventregistry.org/api/v1/article/getArticles"
        self.seen_articles: set = set()
        self.last_query_time: Optional[datetime] = None
        self.query_interval_hours = 6  # Only query every 6 hours to conserve tokens
        self.tokens_used = 0
        self.max_tokens = 2000  # Free tier limit
    
    def collect(self, max_articles: int = 5) -> List[MarketInsight]:
        """
        Collect forex-related news articles from NewsAPI.ai
        Very conservative to preserve tokens (2000 total for free tier)
        """
        if not self.api_key or not REQUESTS_AVAILABLE:
            logger.debug("NewsAPI.ai not configured, skipping collection")
            return []
        
        # Check if we should query (rate limit to preserve tokens)
        if self.last_query_time:
            hours_since_last = (datetime.utcnow() - self.last_query_time).total_seconds() / 3600
            if hours_since_last < self.query_interval_hours:
                logger.debug(f"NewsAPI.ai: Skipping query, only {hours_since_last:.1f}h since last (need {self.query_interval_hours}h)")
                return []
        
        # Check token budget
        if self.tokens_used >= self.max_tokens - 10:
            logger.warning(f"NewsAPI.ai: Token budget nearly exhausted ({self.tokens_used}/{self.max_tokens})")
            return []
        
        insights = []
        
        try:
            # Single focused query for forex news to minimize token usage
            articles = self._search_news("forex currency exchange rate central bank", max_articles)
            insights.extend(articles)
            self.last_query_time = datetime.utcnow()
            self.tokens_used += 1
            logger.info(f"NewsAPI.ai: Query successful, tokens used: {self.tokens_used}/{self.max_tokens}")
        except Exception as e:
            logger.warning(f"Error searching NewsAPI.ai: {e}")
        
        return insights
    
    def _search_news(self, query: str, limit: int) -> List[MarketInsight]:
        """Search for news articles using NewsAPI.ai Event Registry API"""
        insights = []
        
        try:
            # NewsAPI.ai uses a different API structure
            payload = {
                "apiKey": self.api_key,
                "keyword": query,
                "lang": "eng",
                "articlesSortBy": "date",
                "articlesCount": limit,
                "includeArticleBody": False,  # Save tokens by not fetching full body
                "includeArticleConcepts": False,
                "includeArticleCategories": False,
                "includeArticleImage": False,
            }
            
            response = requests.post(self.base_url, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            articles = data.get('articles', {}).get('results', [])
            
            for article in articles:
                article_id = hashlib.md5(
                    (article.get('title', '') + article.get('url', '')).encode()
                ).hexdigest()
                
                if article_id in self.seen_articles:
                    continue
                
                self.seen_articles.add(article_id)
                
                title = article.get('title', '')
                description = article.get('body', '')[:300] if article.get('body') else ''
                content = f"{title}. {description}"
                
                score, label = self.sentiment.analyze(content)
                symbols = self._extract_symbols(content)
                
                # Parse timestamp
                try:
                    date_str = article.get('dateTime', article.get('date', ''))
                    if date_str:
                        timestamp = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.utcnow()
                except:
                    timestamp = datetime.utcnow()
                
                source_name = article.get('source', {}).get('title', 'newsapi.ai')
                
                insight = MarketInsight(
                    source=f"newsapi_ai_{source_name}",
                    title=title,
                    content=description[:500] if description else title,
                    sentiment_score=score,
                    sentiment_label=label,
                    relevance_score=0.7 if symbols else 0.5,
                    symbols_mentioned=symbols,
                    timestamp=timestamp,
                    url=article.get('url'),
                    insight_type="news",
                    impact_level="medium"
                )
                
                insights.append(insight)
                
        except Exception as e:
            logger.error(f"NewsAPI.ai error: {e}")
        
        return insights
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract Forex symbols from text"""
        symbols = []
        text_upper = text.upper()
        
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']
        currencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'DOLLAR', 'EURO', 'YEN', 'POUND']
        
        for pair in forex_pairs:
            if pair in text_upper or pair[:3] + '/' + pair[3:] in text_upper:
                if pair not in symbols:
                    symbols.append(pair)
        
        # Also detect currency mentions
        for currency in currencies:
            if currency in text_upper:
                if currency in ['DOLLAR', 'USD'] and 'EURUSD' not in symbols:
                    symbols.append('EURUSD')
                elif currency in ['EURO', 'EUR'] and 'EURUSD' not in symbols:
                    symbols.append('EURUSD')
                elif currency in ['POUND', 'GBP'] and 'GBPUSD' not in symbols:
                    symbols.append('GBPUSD')
                elif currency in ['YEN', 'JPY'] and 'USDJPY' not in symbols:
                    symbols.append('USDJPY')
        
        return symbols
    
    def get_token_usage(self) -> Dict:
        """Get current token usage stats"""
        return {
            'tokens_used': self.tokens_used,
            'tokens_remaining': self.max_tokens - self.tokens_used,
            'max_tokens': self.max_tokens,
            'last_query': self.last_query_time.isoformat() if self.last_query_time else None
        }


class ForexCalendarCollector:
    """Collects economic calendar events"""
    
    def __init__(self):
        self.events_cache: List[Dict] = []
        self.last_fetch: Optional[datetime] = None
    
    def collect(self) -> List[Dict]:
        """Collect upcoming economic events"""
        # Only refresh every hour
        if self.last_fetch and (datetime.utcnow() - self.last_fetch).seconds < 3600:
            return self.events_cache
        
        events = []
        
        # High-impact events to watch for
        high_impact_events = [
            {'name': 'NFP', 'currency': 'USD', 'impact': 'high', 'typical_day': 'first_friday'},
            {'name': 'FOMC', 'currency': 'USD', 'impact': 'high', 'typical_day': 'variable'},
            {'name': 'ECB Rate Decision', 'currency': 'EUR', 'impact': 'high', 'typical_day': 'variable'},
            {'name': 'BOE Rate Decision', 'currency': 'GBP', 'impact': 'high', 'typical_day': 'variable'},
            {'name': 'CPI', 'currency': 'USD', 'impact': 'high', 'typical_day': 'monthly'},
            {'name': 'GDP', 'currency': 'USD', 'impact': 'medium', 'typical_day': 'quarterly'},
            {'name': 'Retail Sales', 'currency': 'USD', 'impact': 'medium', 'typical_day': 'monthly'},
        ]
        
        # Add placeholder events (in production, would fetch from Forex Factory API)
        now = datetime.utcnow()
        for event in high_impact_events:
            events.append({
                'name': event['name'],
                'currency': event['currency'],
                'impact': event['impact'],
                'estimated_time': now + timedelta(days=7),  # Placeholder
                'actual': None,
                'forecast': None,
                'previous': None
            })
        
        self.events_cache = events
        self.last_fetch = datetime.utcnow()
        
        return events


class KnowledgeDistiller:
    """Distills raw insights into actionable trading knowledge"""
    
    def __init__(self):
        self.symbol_sentiment_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.strategy_hints: List[str] = []
    
    def distill(self, insights: List[MarketInsight], events: List[Dict]) -> KnowledgeState:
        """Distill insights into actionable knowledge state"""
        
        # Calculate overall sentiment
        if insights:
            weighted_sentiment = sum(
                i.sentiment_score * i.relevance_score for i in insights
            )
            total_weight = sum(i.relevance_score for i in insights)
            overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        else:
            overall_sentiment = 0
        
        # Calculate sentiment by symbol
        sentiment_by_symbol = self._calculate_symbol_sentiment(insights)
        
        # Extract trending topics
        trending = self._extract_trending_topics(insights)
        
        # Generate strategy hints
        hints = self._generate_strategy_hints(overall_sentiment, sentiment_by_symbol, events)
        
        # Format upcoming events
        upcoming = [
            {
                'name': e['name'],
                'currency': e['currency'],
                'impact': e['impact'],
                'time': e['estimated_time'].isoformat() if isinstance(e['estimated_time'], datetime) else str(e['estimated_time'])
            }
            for e in events[:5]
        ]
        
        return KnowledgeState(
            overall_sentiment=overall_sentiment,
            sentiment_by_symbol=sentiment_by_symbol,
            upcoming_events=upcoming,
            trending_topics=trending,
            strategy_hints=hints,
            last_update=datetime.utcnow(),
            insights_count=len(insights)
        )
    
    def _calculate_symbol_sentiment(self, insights: List[MarketInsight]) -> Dict[str, float]:
        """Calculate sentiment score for each symbol"""
        symbol_scores: Dict[str, List[float]] = defaultdict(list)
        
        for insight in insights:
            for symbol in insight.symbols_mentioned:
                symbol_scores[symbol].append(insight.sentiment_score * insight.relevance_score)
        
        return {
            symbol: sum(scores) / len(scores) if scores else 0
            for symbol, scores in symbol_scores.items()
        }
    
    def _extract_trending_topics(self, insights: List[MarketInsight]) -> List[str]:
        """Extract trending topics from insights"""
        topic_counts: Dict[str, int] = defaultdict(int)
        
        keywords = [
            'inflation', 'interest rate', 'fed', 'ecb', 'recession',
            'employment', 'gdp', 'trade war', 'tariffs', 'stimulus',
            'hawkish', 'dovish', 'risk-on', 'risk-off'
        ]
        
        for insight in insights:
            content_lower = insight.content.lower()
            for keyword in keywords:
                if keyword in content_lower:
                    topic_counts[keyword] += 1
        
        # Return top 5 trending topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:5]]
    
    def _generate_strategy_hints(
        self, 
        overall_sentiment: float, 
        symbol_sentiment: Dict[str, float],
        events: List[Dict]
    ) -> List[str]:
        """Generate actionable strategy hints"""
        hints = []
        
        # Overall market sentiment hint
        if overall_sentiment > 0.3:
            hints.append("Strong bullish sentiment detected - favor long positions on USD pairs")
        elif overall_sentiment < -0.3:
            hints.append("Strong bearish sentiment detected - favor short positions on USD pairs")
        else:
            hints.append("Mixed sentiment - consider range-bound strategies")
        
        # Symbol-specific hints
        for symbol, sentiment in symbol_sentiment.items():
            if sentiment > 0.4:
                hints.append(f"{symbol}: Bullish bias detected - look for long entries on pullbacks")
            elif sentiment < -0.4:
                hints.append(f"{symbol}: Bearish bias detected - look for short entries on rallies")
        
        # Event-based hints
        high_impact_soon = [e for e in events if e.get('impact') == 'high']
        if high_impact_soon:
            hints.append(f"High-impact events upcoming - reduce position sizes and widen stops")
        
        return hints


class KnowledgeAcquisitionEngine:
    """
    Main engine for 24/7 knowledge acquisition.
    Runs continuously to gather and process market intelligence.
    """
    
    def __init__(self, storage_path: str = "./data/knowledge"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.rss_collector = RSSFeedCollector(self.sentiment_analyzer)
        self.reddit_collector = RedditCollector(self.sentiment_analyzer)
        self.news_collector = NewsAPICollector(self.sentiment_analyzer)
        self.calendar_collector = ForexCalendarCollector()
        self.distiller = KnowledgeDistiller()
        
        # State
        self.current_state: Optional[KnowledgeState] = None
        self.all_insights: List[MarketInsight] = []
        self.running = False
        self._thread: Optional[threading.Thread] = None
        
        # Load previous state
        self._load_state()
        
        logger.info("KnowledgeAcquisitionEngine initialized")
    
    def _load_state(self):
        """Load previous knowledge state from disk"""
        state_file = self.storage_path / "knowledge_state.json"
        insights_file = self.storage_path / "insights_history.json"
        
        try:
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.current_state = KnowledgeState(
                        overall_sentiment=data.get('overall_sentiment', 0),
                        sentiment_by_symbol=data.get('sentiment_by_symbol', {}),
                        upcoming_events=data.get('upcoming_events', []),
                        trending_topics=data.get('trending_topics', []),
                        strategy_hints=data.get('strategy_hints', []),
                        last_update=datetime.fromisoformat(data['last_update']) if data.get('last_update') else None,
                        insights_count=data.get('insights_count', 0)
                    )
                    logger.info(f"Loaded knowledge state from {state_file}")
        except Exception as e:
            logger.warning(f"Could not load knowledge state: {e}")
        
        try:
            if insights_file.exists():
                with open(insights_file, 'r') as f:
                    data = json.load(f)
                    # Only load recent insights (last 24 hours)
                    cutoff = datetime.utcnow() - timedelta(hours=24)
                    for item in data[-100:]:  # Last 100 insights
                        try:
                            timestamp = datetime.fromisoformat(item['timestamp'])
                            if timestamp > cutoff:
                                insight = MarketInsight(
                                    source=item['source'],
                                    title=item['title'],
                                    content=item['content'],
                                    sentiment_score=item['sentiment_score'],
                                    sentiment_label=item['sentiment_label'],
                                    relevance_score=item['relevance_score'],
                                    symbols_mentioned=item['symbols_mentioned'],
                                    timestamp=timestamp,
                                    url=item.get('url'),
                                    impact_level=item.get('impact_level', 'low'),
                                    insight_type=item.get('insight_type', 'general')
                                )
                                self.all_insights.append(insight)
                        except:
                            pass
                    logger.info(f"Loaded {len(self.all_insights)} recent insights")
        except Exception as e:
            logger.warning(f"Could not load insights history: {e}")
    
    def _save_state(self):
        """Save current knowledge state to disk"""
        state_file = self.storage_path / "knowledge_state.json"
        insights_file = self.storage_path / "insights_history.json"
        
        try:
            if self.current_state:
                with open(state_file, 'w') as f:
                    json.dump(self.current_state.to_dict(), f, indent=2)
            
            # Save recent insights
            recent_insights = [i.to_dict() for i in self.all_insights[-200:]]
            with open(insights_file, 'w') as f:
                json.dump(recent_insights, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving knowledge state: {e}")
    
    def collect_all(self) -> KnowledgeState:
        """Collect insights from all sources and distill into knowledge state"""
        logger.info("Starting knowledge collection cycle...")
        
        all_insights = []
        
        # Collect from RSS feeds
        try:
            rss_insights = self.rss_collector.collect(max_articles=30)
            all_insights.extend(rss_insights)
            logger.info(f"Collected {len(rss_insights)} RSS insights")
        except Exception as e:
            logger.error(f"RSS collection error: {e}")
        
        # Collect from Reddit
        try:
            reddit_insights = self.reddit_collector.collect(max_posts=20)
            all_insights.extend(reddit_insights)
            logger.info(f"Collected {len(reddit_insights)} Reddit insights")
        except Exception as e:
            logger.error(f"Reddit collection error: {e}")
        
        # Collect from NewsAPI
        try:
            news_insights = self.news_collector.collect(max_articles=15)
            all_insights.extend(news_insights)
            logger.info(f"Collected {len(news_insights)} NewsAPI insights")
        except Exception as e:
            logger.error(f"NewsAPI collection error: {e}")
        
        # Collect calendar events
        try:
            events = self.calendar_collector.collect()
            logger.info(f"Collected {len(events)} calendar events")
        except Exception as e:
            logger.error(f"Calendar collection error: {e}")
            events = []
        
        # Add to history
        self.all_insights.extend(all_insights)
        
        # Keep only last 500 insights
        if len(self.all_insights) > 500:
            self.all_insights = self.all_insights[-500:]
        
        # Distill into knowledge state
        self.current_state = self.distiller.distill(all_insights, events)
        
        # Save state
        self._save_state()
        
        logger.info(f"Knowledge collection complete: {len(all_insights)} new insights, "
                   f"overall sentiment: {self.current_state.overall_sentiment:.2f}")
        
        return self.current_state
    
    def get_current_state(self) -> Optional[KnowledgeState]:
        """Get current knowledge state"""
        return self.current_state
    
    def get_symbol_sentiment(self, symbol: str) -> float:
        """Get sentiment for a specific symbol"""
        if self.current_state and symbol in self.current_state.sentiment_by_symbol:
            return self.current_state.sentiment_by_symbol[symbol]
        return 0.0
    
    def get_strategy_hints(self) -> List[str]:
        """Get current strategy hints"""
        if self.current_state:
            return self.current_state.strategy_hints
        return []
    
    def should_avoid_trading(self) -> Tuple[bool, str]:
        """Check if trading should be avoided due to upcoming events"""
        if not self.current_state:
            return False, ""
        
        for event in self.current_state.upcoming_events:
            if event.get('impact') == 'high':
                return True, f"High-impact event upcoming: {event.get('name')}"
        
        return False, ""
    
    def start_background_collection(self, interval_minutes: int = 30):
        """Start background collection thread"""
        if self.running:
            logger.warning("Background collection already running")
            return
        
        self.running = True
        
        def collection_loop():
            while self.running:
                try:
                    self.collect_all()
                except Exception as e:
                    logger.error(f"Background collection error: {e}")
                
                # Sleep for interval
                for _ in range(interval_minutes * 60):
                    if not self.running:
                        break
                    time.sleep(1)
        
        self._thread = threading.Thread(target=collection_loop, daemon=True)
        self._thread.start()
        logger.info(f"Background knowledge collection started (interval: {interval_minutes} min)")
    
    def stop_background_collection(self):
        """Stop background collection thread"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Background knowledge collection stopped")
    
    def generate_learning_report(self) -> str:
        """Generate a human-readable learning report"""
        if not self.current_state:
            return "No knowledge state available yet."
        
        report = []
        report.append("=" * 60)
        report.append("PHOENIX KNOWLEDGE ACQUISITION REPORT")
        report.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        report.append("=" * 60)
        
        report.append(f"\nOverall Market Sentiment: {self.current_state.overall_sentiment:.2f}")
        if self.current_state.overall_sentiment > 0.2:
            report.append("  -> BULLISH bias detected")
        elif self.current_state.overall_sentiment < -0.2:
            report.append("  -> BEARISH bias detected")
        else:
            report.append("  -> NEUTRAL/MIXED sentiment")
        
        report.append(f"\nInsights Analyzed: {self.current_state.insights_count}")
        
        if self.current_state.sentiment_by_symbol:
            report.append("\nSentiment by Symbol:")
            for symbol, sentiment in sorted(
                self.current_state.sentiment_by_symbol.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ):
                direction = "BULLISH" if sentiment > 0 else "BEARISH" if sentiment < 0 else "NEUTRAL"
                report.append(f"  {symbol}: {sentiment:+.2f} ({direction})")
        
        if self.current_state.trending_topics:
            report.append("\nTrending Topics:")
            for topic in self.current_state.trending_topics:
                report.append(f"  - {topic}")
        
        if self.current_state.strategy_hints:
            report.append("\nStrategy Hints:")
            for hint in self.current_state.strategy_hints:
                report.append(f"  * {hint}")
        
        if self.current_state.upcoming_events:
            report.append("\nUpcoming High-Impact Events:")
            for event in self.current_state.upcoming_events:
                report.append(f"  - {event.get('name')} ({event.get('currency')}) - {event.get('impact')} impact")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Global instance for easy access
knowledge_engine = KnowledgeAcquisitionEngine()


if __name__ == "__main__":
    # Test the knowledge acquisition engine
    logging.basicConfig(level=logging.INFO)
    
    engine = KnowledgeAcquisitionEngine()
    state = engine.collect_all()
    
    print(engine.generate_learning_report())
