"""
AWS Bedrock AI Integration - Claude-Powered Market Analysis
Provides institutional-grade AI analysis for trading decisions
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BEDROCK_AVAILABLE = False
bedrock_client = None

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BEDROCK_AVAILABLE = True
except ImportError:
    logger.warning("boto3 not available - Bedrock AI features disabled")


@dataclass
class AIAnalysis:
    """AI-generated market analysis"""
    timestamp: datetime
    symbol: str
    analysis_type: str
    sentiment: str
    confidence: float
    recommendation: str
    reasoning: str
    risk_factors: List[str]
    entry_suggestion: Optional[str]
    exit_suggestion: Optional[str]
    position_size_modifier: float
    raw_response: str


class BedrockAI:
    """
    AWS Bedrock integration for AI-powered market analysis.
    Uses Claude for intelligent market interpretation and decision support.
    """
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.client = None
        self.model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        self.fallback_model_id = 'anthropic.claude-instant-v1'
        self.is_available = False
        self._initialize_client()
        
        self.analysis_cache: Dict[str, AIAnalysis] = {}
        self.cache_ttl_seconds = 300
    
    def _initialize_client(self):
        """Initialize Bedrock client"""
        if not BEDROCK_AVAILABLE:
            logger.warning("Bedrock not available - boto3 not installed")
            return
        
        try:
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=self.region
            )
            self.client.list_foundation_models = lambda: None
            self.is_available = True
            logger.info("Bedrock AI client initialized successfully")
        except NoCredentialsError:
            logger.warning("AWS credentials not found - Bedrock AI disabled")
            self.is_available = False
        except Exception as e:
            logger.warning(f"Could not initialize Bedrock client: {e}")
            self.is_available = False
    
    def analyze_market(self, symbol: str, market_data: Dict, 
                       indicators: Dict, regime: str,
                       recent_patterns: List[str] = None) -> AIAnalysis:
        """
        Get AI analysis of current market conditions.
        Like having a veteran trader analyze the market for you.
        """
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        if not self.is_available:
            return self._fallback_analysis(symbol, market_data, indicators, regime)
        
        prompt = self._build_analysis_prompt(symbol, market_data, indicators, regime, recent_patterns)
        
        try:
            response = self._invoke_claude(prompt)
            analysis = self._parse_analysis_response(response, symbol)
            self.analysis_cache[cache_key] = analysis
            return analysis
        except Exception as e:
            logger.error(f"Bedrock analysis failed: {e}")
            return self._fallback_analysis(symbol, market_data, indicators, regime)
    
    def _build_analysis_prompt(self, symbol: str, market_data: Dict,
                               indicators: Dict, regime: str,
                               patterns: List[str] = None) -> str:
        """Build prompt for Claude analysis"""
        prompt = f"""You are an expert Forex trader with 30 years of experience. Analyze the following market data and provide a trading recommendation.

SYMBOL: {symbol}
CURRENT REGIME: {regime}

MARKET DATA:
- Current Price: {market_data.get('close', 'N/A')}
- Open: {market_data.get('open', 'N/A')}
- High: {market_data.get('high', 'N/A')}
- Low: {market_data.get('low', 'N/A')}
- Volume: {market_data.get('volume', 'N/A')}

TECHNICAL INDICATORS:
- RSI (14): {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')}
- MACD Signal: {indicators.get('macd_signal', 'N/A')}
- ADX: {indicators.get('adx', 'N/A')}
- ATR: {indicators.get('atr', 'N/A')}
- EMA 20: {indicators.get('ema_20', 'N/A')}
- EMA 50: {indicators.get('ema_50', 'N/A')}
- Bollinger Upper: {indicators.get('bb_upper', 'N/A')}
- Bollinger Lower: {indicators.get('bb_lower', 'N/A')}

DISCOVERED PATTERNS:
{chr(10).join(patterns) if patterns else 'No significant patterns detected'}

Provide your analysis in the following JSON format:
{{
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": 0.0-1.0,
    "recommendation": "buy" | "sell" | "hold" | "wait",
    "reasoning": "Your detailed reasoning",
    "risk_factors": ["list", "of", "risks"],
    "entry_suggestion": "Specific entry level or null",
    "exit_suggestion": "Specific exit level or null",
    "position_size_modifier": 0.5-1.5
}}

Be conservative and prioritize capital preservation. Only recommend trades with clear setups."""
        
        return prompt
    
    def _invoke_claude(self, prompt: str) -> str:
        """Invoke Claude model via Bedrock"""
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3
        })
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        except ClientError as e:
            if 'AccessDeniedException' in str(e):
                logger.warning("No access to Claude Sonnet, trying fallback model")
                return self._invoke_fallback_model(prompt)
            raise
    
    def _invoke_fallback_model(self, prompt: str) -> str:
        """Use fallback model if primary is unavailable"""
        body = json.dumps({
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": 1024,
            "temperature": 0.3
        })
        
        response = self.client.invoke_model(
            modelId=self.fallback_model_id,
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['completion']
    
    def _parse_analysis_response(self, response: str, symbol: str) -> AIAnalysis:
        """Parse Claude's response into structured analysis"""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
            
            return AIAnalysis(
                timestamp=datetime.now(),
                symbol=symbol,
                analysis_type='bedrock_claude',
                sentiment=data.get('sentiment', 'neutral'),
                confidence=float(data.get('confidence', 0.5)),
                recommendation=data.get('recommendation', 'hold'),
                reasoning=data.get('reasoning', ''),
                risk_factors=data.get('risk_factors', []),
                entry_suggestion=data.get('entry_suggestion'),
                exit_suggestion=data.get('exit_suggestion'),
                position_size_modifier=float(data.get('position_size_modifier', 1.0)),
                raw_response=response
            )
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            return self._fallback_analysis(symbol, {}, {}, 'unknown')
    
    def _fallback_analysis(self, symbol: str, market_data: Dict,
                          indicators: Dict, regime: str) -> AIAnalysis:
        """Provide rule-based analysis when AI is unavailable"""
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 25)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        
        if rsi < 30:
            sentiment = 'bullish'
            recommendation = 'buy'
            reasoning = f"RSI oversold at {rsi:.1f}, potential reversal"
            confidence = 0.6
        elif rsi > 70:
            sentiment = 'bearish'
            recommendation = 'sell'
            reasoning = f"RSI overbought at {rsi:.1f}, potential reversal"
            confidence = 0.6
        elif adx > 40 and macd > macd_signal:
            sentiment = 'bullish'
            recommendation = 'buy'
            reasoning = f"Strong trend (ADX={adx:.1f}) with bullish MACD crossover"
            confidence = 0.7
        elif adx > 40 and macd < macd_signal:
            sentiment = 'bearish'
            recommendation = 'sell'
            reasoning = f"Strong trend (ADX={adx:.1f}) with bearish MACD crossover"
            confidence = 0.7
        else:
            sentiment = 'neutral'
            recommendation = 'hold'
            reasoning = "No clear setup - waiting for better opportunity"
            confidence = 0.5
        
        risk_factors = []
        if regime == 'high_vol':
            risk_factors.append("High volatility regime - use smaller position size")
            confidence *= 0.8
        if adx < 20:
            risk_factors.append("Weak trend - choppy conditions likely")
        
        return AIAnalysis(
            timestamp=datetime.now(),
            symbol=symbol,
            analysis_type='rule_based_fallback',
            sentiment=sentiment,
            confidence=confidence,
            recommendation=recommendation,
            reasoning=reasoning,
            risk_factors=risk_factors,
            entry_suggestion=None,
            exit_suggestion=None,
            position_size_modifier=0.8 if regime == 'high_vol' else 1.0,
            raw_response="Fallback analysis - Bedrock unavailable"
        )
    
    def analyze_news_sentiment(self, news_items: List[Dict]) -> Dict[str, Any]:
        """Analyze news sentiment using AI"""
        if not news_items:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'impact': 'low'}
        
        if not self.is_available:
            return self._fallback_news_analysis(news_items)
        
        headlines = [item.get('title', '') for item in news_items[:10]]
        
        prompt = f"""Analyze these Forex-related news headlines and provide market sentiment:

HEADLINES:
{chr(10).join(f'- {h}' for h in headlines)}

Respond in JSON format:
{{
    "overall_sentiment": "bullish" | "bearish" | "neutral",
    "confidence": 0.0-1.0,
    "impact": "high" | "medium" | "low",
    "key_themes": ["list", "of", "themes"],
    "affected_pairs": ["EUR/USD", "GBP/USD", etc],
    "summary": "Brief summary"
}}"""
        
        try:
            response = self._invoke_claude(prompt)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except Exception as e:
            logger.error(f"News analysis failed: {e}")
        
        return self._fallback_news_analysis(news_items)
    
    def _fallback_news_analysis(self, news_items: List[Dict]) -> Dict[str, Any]:
        """Simple keyword-based news analysis"""
        bullish_keywords = ['growth', 'rise', 'gain', 'bullish', 'rally', 'surge', 'strong']
        bearish_keywords = ['fall', 'drop', 'decline', 'bearish', 'crash', 'weak', 'recession']
        
        bullish_count = 0
        bearish_count = 0
        
        for item in news_items:
            text = (item.get('title', '') + ' ' + item.get('description', '')).lower()
            bullish_count += sum(1 for kw in bullish_keywords if kw in text)
            bearish_count += sum(1 for kw in bearish_keywords if kw in text)
        
        if bullish_count > bearish_count * 1.5:
            sentiment = 'bullish'
        elif bearish_count > bullish_count * 1.5:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'overall_sentiment': sentiment,
            'confidence': 0.5,
            'impact': 'medium',
            'key_themes': [],
            'affected_pairs': [],
            'summary': 'Keyword-based analysis (AI unavailable)'
        }
    
    def get_trade_explanation(self, trade_data: Dict, outcome: str) -> str:
        """Get AI explanation for a trade outcome"""
        if not self.is_available:
            return f"Trade {outcome}. Review indicators and market conditions for insights."
        
        prompt = f"""Explain why this Forex trade was {outcome}:

TRADE DETAILS:
- Symbol: {trade_data.get('symbol')}
- Direction: {trade_data.get('direction')}
- Entry: {trade_data.get('entry_price')}
- Exit: {trade_data.get('exit_price')}
- P/L: {trade_data.get('profit')}

INDICATORS AT ENTRY:
{json.dumps(trade_data.get('indicators', {}), indent=2)}

REGIME: {trade_data.get('regime')}

Provide a brief, actionable explanation (2-3 sentences) of what worked or didn't work."""
        
        try:
            response = self._invoke_claude(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Trade explanation failed: {e}")
            return f"Trade {outcome}. Unable to generate AI explanation."
    
    def generate_daily_insights(self, daily_stats: Dict, 
                                patterns: List[str]) -> List[str]:
        """Generate AI-powered daily trading insights"""
        if not self.is_available:
            return self._fallback_daily_insights(daily_stats)
        
        prompt = f"""Based on today's trading performance, provide 3-5 actionable insights:

DAILY STATISTICS:
- Total Trades: {daily_stats.get('total_trades', 0)}
- Win Rate: {daily_stats.get('win_rate', 0):.1%}
- Total P/L: ${daily_stats.get('total_profit', 0):.2f}
- Max Drawdown: {daily_stats.get('max_drawdown', 0):.1%}
- Best Trade: ${daily_stats.get('best_trade', 0):.2f}
- Worst Trade: ${daily_stats.get('worst_trade', 0):.2f}

DISCOVERED PATTERNS:
{chr(10).join(patterns) if patterns else 'No significant patterns'}

Provide insights as a JSON array of strings:
["insight 1", "insight 2", ...]"""
        
        try:
            response = self._invoke_claude(prompt)
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except Exception as e:
            logger.error(f"Daily insights generation failed: {e}")
        
        return self._fallback_daily_insights(daily_stats)
    
    def _fallback_daily_insights(self, daily_stats: Dict) -> List[str]:
        """Generate rule-based daily insights"""
        insights = []
        
        win_rate = daily_stats.get('win_rate', 0)
        if win_rate > 0.7:
            insights.append("Excellent win rate today - maintain current strategy discipline")
        elif win_rate < 0.4:
            insights.append("Low win rate - review entry criteria and consider tighter filters")
        
        max_dd = daily_stats.get('max_drawdown', 0)
        if max_dd > 0.1:
            insights.append("Significant drawdown detected - consider reducing position sizes")
        
        total_trades = daily_stats.get('total_trades', 0)
        if total_trades > 20:
            insights.append("High trade frequency - ensure quality over quantity")
        elif total_trades < 3:
            insights.append("Low trade count - review if opportunities were missed")
        
        if not insights:
            insights.append("Trading performance within normal parameters")
        
        return insights


bedrock_ai = BedrockAI()
