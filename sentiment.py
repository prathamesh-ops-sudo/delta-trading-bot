"""
Sentiment Analysis Module
Analyzes news, social media, and market sentiment for trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import re
import json
import os

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available - using basic sentiment analysis")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

from config import config, NEWS_IMPACT

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Sentiment score for a piece of content"""
    source: str
    content: str
    timestamp: datetime
    compound: float  # -1 to 1
    positive: float
    negative: float
    neutral: float
    relevance: float  # 0 to 1, how relevant to forex
    impact: str  # 'high', 'medium', 'low'
    currencies_mentioned: List[str] = field(default_factory=list)


@dataclass
class MarketSentiment:
    """Aggregated market sentiment"""
    timestamp: datetime
    overall_score: float  # -1 to 1
    bullish_pct: float
    bearish_pct: float
    neutral_pct: float
    confidence: float
    num_sources: int
    currency_sentiments: Dict[str, float] = field(default_factory=dict)
    top_themes: List[str] = field(default_factory=list)


class ForexLexicon:
    """Custom forex-specific sentiment lexicon"""
    
    BULLISH_TERMS = {
        'rally': 2.0, 'surge': 2.0, 'soar': 2.0, 'jump': 1.5, 'gain': 1.0,
        'rise': 1.0, 'climb': 1.0, 'advance': 1.0, 'strengthen': 1.5,
        'bullish': 2.0, 'hawkish': 1.5, 'optimistic': 1.0, 'recovery': 1.0,
        'growth': 1.0, 'expansion': 1.0, 'boom': 2.0, 'breakout': 1.5,
        'support': 0.5, 'buy': 1.0, 'long': 0.5, 'uptrend': 1.5,
        'higher': 0.5, 'strong': 1.0, 'robust': 1.0, 'positive': 1.0,
        'beat': 1.0, 'exceed': 1.0, 'outperform': 1.5, 'upgrade': 1.5,
        'rate hike': 1.5, 'tightening': 1.0, 'inflation': -0.5
    }
    
    BEARISH_TERMS = {
        'crash': -2.0, 'plunge': -2.0, 'tumble': -1.5, 'drop': -1.0,
        'fall': -1.0, 'decline': -1.0, 'slide': -1.0, 'weaken': -1.5,
        'bearish': -2.0, 'dovish': -1.5, 'pessimistic': -1.0, 'recession': -2.0,
        'contraction': -1.5, 'slowdown': -1.0, 'crisis': -2.0, 'breakdown': -1.5,
        'resistance': -0.5, 'sell': -1.0, 'short': -0.5, 'downtrend': -1.5,
        'lower': -0.5, 'weak': -1.0, 'poor': -1.0, 'negative': -1.0,
        'miss': -1.0, 'disappoint': -1.0, 'underperform': -1.5, 'downgrade': -1.5,
        'rate cut': -1.5, 'easing': -1.0, 'deflation': -1.0, 'default': -2.0
    }
    
    CURRENCY_KEYWORDS = {
        'USD': ['dollar', 'usd', 'greenback', 'buck', 'fed', 'federal reserve', 'us economy', 'american'],
        'EUR': ['euro', 'eur', 'ecb', 'european central bank', 'eurozone', 'eu economy'],
        'GBP': ['pound', 'gbp', 'sterling', 'boe', 'bank of england', 'uk economy', 'british'],
        'JPY': ['yen', 'jpy', 'boj', 'bank of japan', 'japanese', 'japan economy'],
        'CHF': ['franc', 'chf', 'swiss', 'snb', 'swiss national bank'],
        'AUD': ['aussie', 'aud', 'rba', 'reserve bank of australia', 'australian'],
        'CAD': ['loonie', 'cad', 'boc', 'bank of canada', 'canadian'],
        'NZD': ['kiwi', 'nzd', 'rbnz', 'new zealand']
    }
    
    HIGH_IMPACT_EVENTS = [
        'nfp', 'non-farm', 'payroll', 'fomc', 'rate decision', 'gdp',
        'inflation', 'cpi', 'ppi', 'employment', 'unemployment', 'retail sales',
        'trade balance', 'pmi', 'manufacturing', 'services', 'housing',
        'consumer confidence', 'central bank', 'monetary policy', 'quantitative'
    ]
    
    @classmethod
    def get_forex_sentiment(cls, text: str) -> Tuple[float, List[str]]:
        """Calculate forex-specific sentiment score"""
        text_lower = text.lower()
        
        score = 0.0
        currencies = []
        
        # Check bullish terms
        for term, weight in cls.BULLISH_TERMS.items():
            if term in text_lower:
                score += weight
        
        # Check bearish terms
        for term, weight in cls.BEARISH_TERMS.items():
            if term in text_lower:
                score += weight
        
        # Identify currencies mentioned
        for currency, keywords in cls.CURRENCY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if currency not in currencies:
                        currencies.append(currency)
                    break
        
        # Normalize score
        score = np.clip(score / 5, -1, 1)
        
        return score, currencies
    
    @classmethod
    def get_impact_level(cls, text: str) -> str:
        """Determine impact level of news"""
        text_lower = text.lower()
        
        high_impact_count = sum(1 for event in cls.HIGH_IMPACT_EVENTS if event in text_lower)
        
        if high_impact_count >= 2:
            return 'high'
        elif high_impact_count == 1:
            return 'medium'
        else:
            return 'low'


class VADERSentimentAnalyzer:
    """VADER-based sentiment analyzer with forex enhancements"""
    
    def __init__(self):
        if NLTK_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
            # Add forex-specific terms to lexicon
            self._enhance_lexicon()
        else:
            self.analyzer = None
    
    def _enhance_lexicon(self):
        """Add forex-specific terms to VADER lexicon"""
        if self.analyzer is None:
            return
        
        # Add bullish terms
        for term, score in ForexLexicon.BULLISH_TERMS.items():
            self.analyzer.lexicon[term] = score
        
        # Add bearish terms
        for term, score in ForexLexicon.BEARISH_TERMS.items():
            self.analyzer.lexicon[term] = score
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        if self.analyzer is None:
            return self._basic_analyze(text)
        
        try:
            scores = self.analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except Exception as e:
            logger.warning(f"VADER analysis failed: {e}")
            return self._basic_analyze(text)
    
    def _basic_analyze(self, text: str) -> Dict[str, float]:
        """Basic sentiment analysis fallback"""
        forex_score, _ = ForexLexicon.get_forex_sentiment(text)
        
        if forex_score > 0:
            return {
                'compound': forex_score,
                'positive': abs(forex_score),
                'negative': 0,
                'neutral': 1 - abs(forex_score)
            }
        elif forex_score < 0:
            return {
                'compound': forex_score,
                'positive': 0,
                'negative': abs(forex_score),
                'neutral': 1 - abs(forex_score)
            }
        else:
            return {
                'compound': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 1
            }


class TextBlobAnalyzer:
    """TextBlob-based sentiment analyzer"""
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        if not TEXTBLOB_AVAILABLE:
            return {'polarity': 0, 'subjectivity': 0.5}
        
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
            return {'polarity': 0, 'subjectivity': 0.5}


class NewsSentimentAnalyzer:
    """Analyzes news articles for forex sentiment"""
    
    def __init__(self):
        self.vader = VADERSentimentAnalyzer()
        self.textblob = TextBlobAnalyzer()
        self.sentiment_cache: Dict[str, SentimentScore] = {}
    
    def analyze_article(self, title: str, content: str, source: str,
                        timestamp: datetime = None) -> SentimentScore:
        """Analyze a news article"""
        timestamp = timestamp or datetime.now()
        
        # Combine title and content (title weighted more)
        full_text = f"{title} {title} {content}"
        
        # VADER analysis
        vader_scores = self.vader.analyze(full_text)
        
        # TextBlob analysis
        textblob_scores = self.textblob.analyze(full_text)
        
        # Forex-specific analysis
        forex_score, currencies = ForexLexicon.get_forex_sentiment(full_text)
        
        # Combine scores (weighted average)
        compound = (
            vader_scores['compound'] * 0.4 +
            textblob_scores.get('polarity', 0) * 0.3 +
            forex_score * 0.3
        )
        
        # Calculate relevance
        relevance = self._calculate_relevance(full_text, currencies)
        
        # Get impact level
        impact = ForexLexicon.get_impact_level(full_text)
        
        return SentimentScore(
            source=source,
            content=title[:200],
            timestamp=timestamp,
            compound=compound,
            positive=vader_scores['positive'],
            negative=vader_scores['negative'],
            neutral=vader_scores['neutral'],
            relevance=relevance,
            impact=impact,
            currencies_mentioned=currencies
        )
    
    def _calculate_relevance(self, text: str, currencies: List[str]) -> float:
        """Calculate how relevant the text is to forex trading"""
        relevance = 0.0
        
        # Currency mentions
        relevance += min(len(currencies) * 0.2, 0.6)
        
        # Forex keywords
        forex_keywords = ['forex', 'fx', 'currency', 'exchange rate', 'central bank',
                         'interest rate', 'monetary policy', 'trade', 'economy']
        
        text_lower = text.lower()
        keyword_count = sum(1 for kw in forex_keywords if kw in text_lower)
        relevance += min(keyword_count * 0.1, 0.4)
        
        return min(relevance, 1.0)
    
    def analyze_batch(self, articles: List[Dict]) -> List[SentimentScore]:
        """Analyze multiple articles"""
        scores = []
        for article in articles:
            score = self.analyze_article(
                title=article.get('title', ''),
                content=article.get('content', article.get('description', '')),
                source=article.get('source', 'unknown'),
                timestamp=article.get('published_at')
            )
            scores.append(score)
        return scores


class SocialSentimentAnalyzer:
    """Analyzes social media sentiment (Twitter/X, Reddit)"""
    
    def __init__(self):
        self.vader = VADERSentimentAnalyzer()
    
    def analyze_post(self, text: str, source: str, 
                     timestamp: datetime = None) -> SentimentScore:
        """Analyze a social media post"""
        timestamp = timestamp or datetime.now()
        
        # Clean text
        cleaned = self._clean_social_text(text)
        
        # Analyze
        vader_scores = self.vader.analyze(cleaned)
        forex_score, currencies = ForexLexicon.get_forex_sentiment(cleaned)
        
        compound = vader_scores['compound'] * 0.6 + forex_score * 0.4
        relevance = self._calculate_relevance(cleaned, currencies)
        
        return SentimentScore(
            source=source,
            content=text[:200],
            timestamp=timestamp,
            compound=compound,
            positive=vader_scores['positive'],
            negative=vader_scores['negative'],
            neutral=vader_scores['neutral'],
            relevance=relevance,
            impact='low',  # Social media typically lower impact
            currencies_mentioned=currencies
        )
    
    def _clean_social_text(self, text: str) -> str:
        """Clean social media text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags (keep the word)
        text = re.sub(r'#(\w+)', r'\1', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _calculate_relevance(self, text: str, currencies: List[str]) -> float:
        """Calculate relevance for social media"""
        relevance = 0.0
        
        # Currency mentions
        relevance += min(len(currencies) * 0.3, 0.6)
        
        # Trading keywords
        trading_keywords = ['long', 'short', 'buy', 'sell', 'bullish', 'bearish',
                          'support', 'resistance', 'breakout', 'trend']
        
        text_lower = text.lower()
        keyword_count = sum(1 for kw in trading_keywords if kw in text_lower)
        relevance += min(keyword_count * 0.1, 0.4)
        
        return min(relevance, 1.0)


class SentimentAggregator:
    """Aggregates sentiment from multiple sources"""
    
    def __init__(self):
        self.news_analyzer = NewsSentimentAnalyzer()
        self.social_analyzer = SocialSentimentAnalyzer()
        self.sentiment_history: List[MarketSentiment] = []
        
        # Source weights
        self.source_weights = {
            'reuters': 1.5,
            'bloomberg': 1.5,
            'wsj': 1.3,
            'ft': 1.3,
            'cnbc': 1.0,
            'forexlive': 1.2,
            'dailyfx': 1.1,
            'twitter': 0.5,
            'reddit': 0.6,
            'default': 0.8
        }
    
    def aggregate(self, scores: List[SentimentScore]) -> MarketSentiment:
        """Aggregate multiple sentiment scores"""
        if not scores:
            return self._empty_sentiment()
        
        # Filter by relevance
        relevant_scores = [s for s in scores if s.relevance > 0.3]
        
        if not relevant_scores:
            relevant_scores = scores
        
        # Calculate weighted average
        total_weight = 0
        weighted_compound = 0
        currency_scores = {}
        
        for score in relevant_scores:
            # Get source weight
            source_lower = score.source.lower()
            weight = self.source_weights.get(source_lower, self.source_weights['default'])
            
            # Apply relevance and impact multipliers
            weight *= score.relevance
            if score.impact == 'high':
                weight *= 1.5
            elif score.impact == 'medium':
                weight *= 1.2
            
            weighted_compound += score.compound * weight
            total_weight += weight
            
            # Track currency-specific sentiment
            for currency in score.currencies_mentioned:
                if currency not in currency_scores:
                    currency_scores[currency] = {'total': 0, 'weight': 0}
                currency_scores[currency]['total'] += score.compound * weight
                currency_scores[currency]['weight'] += weight
        
        # Calculate final scores
        overall_score = weighted_compound / total_weight if total_weight > 0 else 0
        
        # Calculate percentages
        bullish = len([s for s in relevant_scores if s.compound > 0.1])
        bearish = len([s for s in relevant_scores if s.compound < -0.1])
        neutral = len(relevant_scores) - bullish - bearish
        total = len(relevant_scores)
        
        # Currency sentiments
        currency_sentiments = {}
        for currency, data in currency_scores.items():
            if data['weight'] > 0:
                currency_sentiments[currency] = data['total'] / data['weight']
        
        # Extract top themes
        top_themes = self._extract_themes(relevant_scores)
        
        # Calculate confidence
        confidence = min(1.0, len(relevant_scores) / 10) * (1 - np.std([s.compound for s in relevant_scores]) if len(relevant_scores) > 1 else 0.5)
        
        sentiment = MarketSentiment(
            timestamp=datetime.now(),
            overall_score=overall_score,
            bullish_pct=bullish / total if total > 0 else 0,
            bearish_pct=bearish / total if total > 0 else 0,
            neutral_pct=neutral / total if total > 0 else 0,
            confidence=confidence,
            num_sources=len(relevant_scores),
            currency_sentiments=currency_sentiments,
            top_themes=top_themes
        )
        
        self.sentiment_history.append(sentiment)
        return sentiment
    
    def _empty_sentiment(self) -> MarketSentiment:
        """Return empty sentiment"""
        return MarketSentiment(
            timestamp=datetime.now(),
            overall_score=0,
            bullish_pct=0,
            bearish_pct=0,
            neutral_pct=1,
            confidence=0,
            num_sources=0,
            currency_sentiments={},
            top_themes=[]
        )
    
    def _extract_themes(self, scores: List[SentimentScore]) -> List[str]:
        """Extract common themes from sentiment scores"""
        theme_counts = {}
        
        themes = ['inflation', 'interest rate', 'gdp', 'employment', 'trade',
                 'central bank', 'recession', 'growth', 'policy', 'crisis']
        
        for score in scores:
            content_lower = score.content.lower()
            for theme in themes:
                if theme in content_lower:
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # Sort by count and return top 5
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:5]]
    
    def get_sentiment_for_pair(self, pair: str) -> float:
        """Get sentiment score for a currency pair"""
        if not self.sentiment_history:
            return 0
        
        latest = self.sentiment_history[-1]
        
        base = pair[:3]
        quote = pair[3:]
        
        base_sentiment = latest.currency_sentiments.get(base, 0)
        quote_sentiment = latest.currency_sentiments.get(quote, 0)
        
        # Positive base sentiment or negative quote sentiment = bullish for pair
        return base_sentiment - quote_sentiment
    
    def get_sentiment_trend(self, hours: int = 24) -> Dict:
        """Get sentiment trend over time"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [s for s in self.sentiment_history if s.timestamp > cutoff]
        
        if not recent:
            return {'trend': 'neutral', 'change': 0}
        
        if len(recent) < 2:
            return {'trend': 'neutral', 'change': 0}
        
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_avg = np.mean([s.overall_score for s in first_half])
        second_avg = np.mean([s.overall_score for s in second_half])
        
        change = second_avg - first_avg
        
        if change > 0.1:
            trend = 'improving'
        elif change < -0.1:
            trend = 'deteriorating'
        else:
            trend = 'stable'
        
        return {'trend': trend, 'change': change, 'current': second_avg}


class SentimentManager:
    """Main sentiment management class"""
    
    def __init__(self):
        self.aggregator = SentimentAggregator()
        self.news_analyzer = NewsSentimentAnalyzer()
        self.social_analyzer = SocialSentimentAnalyzer()
        self.current_sentiment: Optional[MarketSentiment] = None
        self.last_update = None
    
    def update_sentiment(self, news_articles: List[Dict] = None,
                         social_posts: List[Dict] = None) -> MarketSentiment:
        """Update market sentiment from all sources"""
        all_scores = []
        
        # Analyze news
        if news_articles:
            news_scores = self.news_analyzer.analyze_batch(news_articles)
            all_scores.extend(news_scores)
        
        # Analyze social media
        if social_posts:
            for post in social_posts:
                score = self.social_analyzer.analyze_post(
                    text=post.get('text', ''),
                    source=post.get('source', 'social'),
                    timestamp=post.get('timestamp')
                )
                all_scores.append(score)
        
        # Aggregate
        self.current_sentiment = self.aggregator.aggregate(all_scores)
        self.last_update = datetime.now()
        
        return self.current_sentiment
    
    def get_trading_signal_adjustment(self, pair: str) -> float:
        """Get sentiment-based adjustment for trading signals"""
        if self.current_sentiment is None:
            return 0
        
        # Get pair-specific sentiment
        pair_sentiment = self.aggregator.get_sentiment_for_pair(pair)
        
        # Get overall sentiment
        overall = self.current_sentiment.overall_score
        
        # Combine (pair-specific weighted more)
        adjustment = pair_sentiment * 0.7 + overall * 0.3
        
        # Scale to reasonable range (-0.2 to 0.2)
        return np.clip(adjustment * 0.2, -0.2, 0.2)
    
    def should_avoid_trading(self) -> Tuple[bool, str]:
        """Check if sentiment suggests avoiding trading"""
        if self.current_sentiment is None:
            return False, ""
        
        # Extreme sentiment
        if abs(self.current_sentiment.overall_score) > 0.8:
            return True, "Extreme market sentiment - high uncertainty"
        
        # Low confidence
        if self.current_sentiment.confidence < 0.2:
            return False, ""  # Not enough data to make decision
        
        # Check for crisis keywords in themes
        crisis_themes = ['crisis', 'crash', 'panic', 'emergency']
        for theme in self.current_sentiment.top_themes:
            if any(crisis in theme.lower() for crisis in crisis_themes):
                return True, f"Crisis-related news detected: {theme}"
        
        return False, ""
    
    def get_sentiment_report(self) -> Dict:
        """Get comprehensive sentiment report"""
        if self.current_sentiment is None:
            return {'status': 'no_data'}
        
        trend = self.aggregator.get_sentiment_trend()
        
        return {
            'timestamp': self.current_sentiment.timestamp.isoformat(),
            'overall_score': self.current_sentiment.overall_score,
            'bullish_pct': self.current_sentiment.bullish_pct,
            'bearish_pct': self.current_sentiment.bearish_pct,
            'neutral_pct': self.current_sentiment.neutral_pct,
            'confidence': self.current_sentiment.confidence,
            'num_sources': self.current_sentiment.num_sources,
            'currency_sentiments': self.current_sentiment.currency_sentiments,
            'top_themes': self.current_sentiment.top_themes,
            'trend': trend,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }


# Singleton instance
sentiment_manager = SentimentManager()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Sentiment Analysis Module...")
    print(f"NLTK available: {NLTK_AVAILABLE}")
    print(f"TextBlob available: {TEXTBLOB_AVAILABLE}")
    
    # Test forex lexicon
    print("\nTesting Forex Lexicon...")
    test_texts = [
        "The dollar rallied strongly after hawkish Fed comments",
        "Euro plunges as ECB signals dovish stance",
        "GBPUSD consolidates near support levels",
        "Risk-off sentiment drives yen higher amid market uncertainty"
    ]
    
    for text in test_texts:
        score, currencies = ForexLexicon.get_forex_sentiment(text)
        impact = ForexLexicon.get_impact_level(text)
        print(f"  '{text[:50]}...'")
        print(f"    Score: {score:.2f}, Currencies: {currencies}, Impact: {impact}")
    
    # Test news analyzer
    print("\nTesting News Sentiment Analyzer...")
    news_analyzer = NewsSentimentAnalyzer()
    
    sample_articles = [
        {
            'title': 'Fed signals aggressive rate hikes ahead',
            'content': 'The Federal Reserve indicated it will continue raising interest rates to combat inflation.',
            'source': 'reuters'
        },
        {
            'title': 'Euro weakens on recession fears',
            'content': 'The euro fell against the dollar as economic data pointed to a potential recession in the eurozone.',
            'source': 'bloomberg'
        }
    ]
    
    for article in sample_articles:
        score = news_analyzer.analyze_article(
            title=article['title'],
            content=article['content'],
            source=article['source']
        )
        print(f"  {article['title']}")
        print(f"    Compound: {score.compound:.2f}, Relevance: {score.relevance:.2f}, "
              f"Impact: {score.impact}, Currencies: {score.currencies_mentioned}")
    
    # Test aggregator
    print("\nTesting Sentiment Aggregator...")
    aggregator = SentimentAggregator()
    
    scores = news_analyzer.analyze_batch(sample_articles)
    market_sentiment = aggregator.aggregate(scores)
    
    print(f"  Overall score: {market_sentiment.overall_score:.2f}")
    print(f"  Bullish: {market_sentiment.bullish_pct:.1%}, Bearish: {market_sentiment.bearish_pct:.1%}")
    print(f"  Confidence: {market_sentiment.confidence:.2f}")
    print(f"  Currency sentiments: {market_sentiment.currency_sentiments}")
    print(f"  Top themes: {market_sentiment.top_themes}")
    
    # Test sentiment manager
    print("\nTesting Sentiment Manager...")
    manager = SentimentManager()
    sentiment = manager.update_sentiment(news_articles=sample_articles)
    
    eurusd_adjustment = manager.get_trading_signal_adjustment('EURUSD')
    print(f"  EURUSD signal adjustment: {eurusd_adjustment:.3f}")
    
    should_avoid, reason = manager.should_avoid_trading()
    print(f"  Should avoid trading: {should_avoid} - {reason}")
    
    report = manager.get_sentiment_report()
    print(f"  Report: {json.dumps(report, indent=2, default=str)}")
    
    print("\nSentiment Analysis Module test complete!")
