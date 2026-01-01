//+------------------------------------------------------------------+
//|                                            TradingBridgeEA.mq5   |
//|                                    AI Trading System Bridge      |
//|                                  Connects to AWS Trading Brain   |
//|                          v3.0 - With Trailing/BE/Partial Close   |
//+------------------------------------------------------------------+
#property copyright "AI Trading System"
#property link      ""
#property version   "3.00"
#property strict

#include <Trade\Trade.mqh>

input string   API_URL = "http://44.213.171.246:5000";
input int      POLL_INTERVAL = 5;
input int      MAGIC_NUMBER = 123456;
input double   MAX_SLIPPAGE = 20;
input bool     ENABLE_TRADING = true;
input int      BARS_TO_SEND = 200;
input string   SYMBOLS_TO_TRACK = "EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,USDCAD,NZDUSD,EURGBP";

// Trade Management Inputs
input bool     ENABLE_BREAKEVEN = true;        // Enable break-even logic
input double   BE_TRIGGER_PIPS = 15;           // Pips profit to trigger break-even
input double   BE_OFFSET_PIPS = 2;             // Pips above entry for BE (to cover spread)
input bool     ENABLE_TRAILING = true;         // Enable trailing stop
input double   TRAIL_START_PIPS = 20;          // Pips profit to start trailing
input double   TRAIL_DISTANCE_PIPS = 15;       // Trail distance in pips
input bool     ENABLE_PARTIAL_CLOSE = true;    // Enable partial close at 1:1 RR
input double   PARTIAL_CLOSE_PERCENT = 50;     // Percent to close at 1:1 RR

CTrade trade;
string lastSignalId = "";
datetime lastBarTime[];
datetime lastDealTime = 0;
bool initialDataSent = false;
string symbolsArray[];

// Position management tracking
struct PositionState {
   ulong ticket;
   bool breakEvenApplied;
   bool partialClosed;
   double originalSL;
   double originalTP;
   double entryPrice;
};
PositionState positionStates[];
int positionStateCount = 0;

int OnInit()
{
   trade.SetExpertMagicNumber(MAGIC_NUMBER);
   trade.SetDeviationInPoints((ulong)MAX_SLIPPAGE);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   StringSplit(SYMBOLS_TO_TRACK, ',', symbolsArray);
   ArrayResize(lastBarTime, ArraySize(symbolsArray));
   ArrayResize(positionStates, 50);  // Pre-allocate for up to 50 positions
   positionStateCount = 0;
   
   for(int i = 0; i < ArraySize(symbolsArray); i++)
   {
      string sym = symbolsArray[i];
      StringTrimLeft(sym);
      StringTrimRight(sym);
      symbolsArray[i] = sym;
      SymbolSelect(sym, true);
      lastBarTime[i] = 0;
   }
   
   Print("TradingBridgeEA v3.0 initialized - Tracking ", ArraySize(symbolsArray), " symbols");
   Print("Trade Management: BE=", ENABLE_BREAKEVEN, " Trail=", ENABLE_TRAILING, " Partial=", ENABLE_PARTIAL_CLOSE);
   RegisterWithAPI();
   EventSetTimer(POLL_INTERVAL);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) { EventKillTimer(); }
void OnTimer()
{
   SendAccountInfo();
   SendMarketData();
   SendClosedTrades();
   PollForSignals();
   ManageOpenPositions();  // Trailing stop, break-even, partial close
   SendPositionUpdates();
}
void OnTick() {}

void RegisterWithAPI()
{
   string url = API_URL + "/api/register";
   string headers = "Content-Type: application/json\r\n";
   string jsonBody = StringFormat(
      "{\"login\":%d,\"server\":\"%s\",\"balance\":%.2f,\"equity\":%.2f,\"leverage\":%d,\"version\":\"2.0\"}",
      AccountInfoInteger(ACCOUNT_LOGIN), AccountInfoString(ACCOUNT_SERVER),
      AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY),
      AccountInfoInteger(ACCOUNT_LEVERAGE));
   char data[], result[]; string resultHeaders;
   StringToCharArray(jsonBody, data, 0, StringLen(jsonBody));
   ArrayResize(data, StringLen(jsonBody));
   int res = WebRequest("POST", url, headers, 5000, data, result, resultHeaders);
   if(res == -1) { Print("Registration failed. Error: ", GetLastError()); }
   else { Print("Registered with API: ", CharArrayToString(result)); }
}

void SendAccountInfo()
{
   string url = API_URL + "/api/account";
   string headers = "Content-Type: application/json\r\n";
   string jsonBody = StringFormat(
      "{\"login\":%d,\"server\":\"%s\",\"balance\":%.2f,\"equity\":%.2f,\"margin\":%.2f,\"free_margin\":%.2f,\"leverage\":%d,\"profit\":%.2f}",
      AccountInfoInteger(ACCOUNT_LOGIN), AccountInfoString(ACCOUNT_SERVER),
      AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY),
      AccountInfoDouble(ACCOUNT_MARGIN), AccountInfoDouble(ACCOUNT_MARGIN_FREE),
      AccountInfoInteger(ACCOUNT_LEVERAGE), AccountInfoDouble(ACCOUNT_PROFIT));
   char data[], result[]; string resultHeaders;
   StringToCharArray(jsonBody, data, 0, StringLen(jsonBody));
   ArrayResize(data, StringLen(jsonBody));
   WebRequest("POST", url, headers, 5000, data, result, resultHeaders);
}

void SendMarketData()
{
   string url = API_URL + "/api/market_data";
   string headers = "Content-Type: application/json\r\n";
   string symbolsJson = "";
   
   for(int i = 0; i < ArraySize(symbolsArray); i++)
   {
      string symbol = symbolsArray[i];
      double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
      double spread = (ask - bid) / SymbolInfoDouble(symbol, SYMBOL_POINT);
      int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      long stopLevel = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
      long freezeLevel = SymbolInfoInteger(symbol, SYMBOL_TRADE_FREEZE_LEVEL);
      double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
      double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
      double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
      
      datetime currentBarTime = iTime(symbol, PERIOD_M1, 0);
      bool sendBars = (!initialDataSent || currentBarTime != lastBarTime[i]);
      
      string barsJson = "[]";
      if(sendBars)
      {
         int barsToSend = initialDataSent ? 5 : BARS_TO_SEND;
         MqlRates rates[];
         int copied = CopyRates(symbol, PERIOD_M1, 0, barsToSend, rates);
         if(copied > 0)
         {
            barsJson = "[";
            for(int j = 0; j < copied; j++)
            {
               if(j > 0) barsJson += ",";
               barsJson += StringFormat("{\"t\":%d,\"o\":%.5f,\"h\":%.5f,\"l\":%.5f,\"c\":%.5f,\"v\":%d}",
                  (long)rates[j].time, rates[j].open, rates[j].high, rates[j].low, rates[j].close, (long)rates[j].tick_volume);
            }
            barsJson += "]";
            lastBarTime[i] = currentBarTime;
         }
      }
      
      if(i > 0) symbolsJson += ",";
      symbolsJson += StringFormat(
         "{\"symbol\":\"%s\",\"bid\":%.5f,\"ask\":%.5f,\"spread\":%.1f,\"digits\":%d,\"point\":%.6f,\"stop_level\":%d,\"freeze_level\":%d,\"tick_value\":%.5f,\"min_lot\":%.2f,\"lot_step\":%.2f,\"bars\":%s}",
         symbol, bid, ask, spread, digits, point, stopLevel, freezeLevel, tickValue, minLot, lotStep, barsJson);
   }
   
   string jsonBody = StringFormat("{\"ts\":%d,\"symbols\":[%s],\"initial\":%s}",
      (long)TimeCurrent(), symbolsJson, initialDataSent ? "false" : "true");
   
   char data[], result[]; string resultHeaders;
   StringToCharArray(jsonBody, data, 0, StringLen(jsonBody));
   ArrayResize(data, StringLen(jsonBody));
   int res = WebRequest("POST", url, headers, 10000, data, result, resultHeaders);
   if(res != -1)
   {
      if(!initialDataSent) { Print("Initial market data sent"); initialDataSent = true; }
      
      // Check if server needs full history resync (e.g., after server restart)
      string response = CharArrayToString(result);
      if(StringFind(response, "\"need_full_history\":true") >= 0)
      {
         Print("Server requested full history resync - resending initial data");
         initialDataSent = false;  // Reset to trigger full resend on next tick
         for(int j = 0; j < ArraySize(lastBarTime); j++) lastBarTime[j] = 0;
      }
   }
}

void SendClosedTrades()
{
   string url = API_URL + "/api/closed_trades";
   string headers = "Content-Type: application/json\r\n";
   datetime fromTime = lastDealTime > 0 ? lastDealTime : TimeCurrent() - 86400;
   if(!HistorySelect(fromTime, TimeCurrent())) return;
   
   int totalDeals = HistoryDealsTotal();
   if(totalDeals == 0) return;
   
   string dealsJson = "";
   int newDeals = 0;
   datetime latestDealTime = lastDealTime;
   
   for(int i = 0; i < totalDeals; i++)
   {
      ulong dealTicket = HistoryDealGetTicket(i);
      if(dealTicket == 0) continue;
      
      ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT) continue;
      
      datetime dealTime = (datetime)HistoryDealGetInteger(dealTicket, DEAL_TIME);
      if(dealTime <= lastDealTime) continue;
      
      long magic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
      if(magic != 0 && magic != MAGIC_NUMBER) continue;
      
      string symbol = HistoryDealGetString(dealTicket, DEAL_SYMBOL);
      ENUM_DEAL_TYPE type = (ENUM_DEAL_TYPE)HistoryDealGetInteger(dealTicket, DEAL_TYPE);
      double volume = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
      double price = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
      double profit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
      double commission = HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);
      double swap = HistoryDealGetDouble(dealTicket, DEAL_SWAP);
      long positionId = HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
      string comment = HistoryDealGetString(dealTicket, DEAL_COMMENT);
      
      if(newDeals > 0) dealsJson += ",";
      dealsJson += StringFormat(
         "{\"ticket\":%d,\"symbol\":\"%s\",\"type\":\"%s\",\"volume\":%.2f,\"price\":%.5f,\"profit\":%.2f,\"commission\":%.2f,\"swap\":%.2f,\"position_id\":%d,\"comment\":\"%s\",\"time\":%d}",
         dealTicket, symbol, type == DEAL_TYPE_BUY ? "buy" : "sell", volume, price, profit, commission, swap, positionId, comment, (long)dealTime);
      
      newDeals++;
      if(dealTime > latestDealTime) latestDealTime = dealTime;
   }
   
   if(newDeals == 0) return;
   
   string jsonBody = StringFormat("{\"deals\":[%s],\"count\":%d}", dealsJson, newDeals);
   char data[], result[]; string resultHeaders;
   StringToCharArray(jsonBody, data, 0, StringLen(jsonBody));
   ArrayResize(data, StringLen(jsonBody));
   int res = WebRequest("POST", url, headers, 5000, data, result, resultHeaders);
   if(res != -1) { Print("Sent ", newDeals, " closed trades for learning"); lastDealTime = latestDealTime; }
}

void PollForSignals()
{
   string url = API_URL + "/api/signals";
   string headers = "Content-Type: application/json\r\n";
   char data[], result[]; string resultHeaders;
   int res = WebRequest("GET", url, headers, 5000, data, result, resultHeaders);
   if(res == -1) return;
   string response = CharArrayToString(result);
   ProcessSignals(response);
}

void ProcessSignals(string jsonResponse)
{
   if(StringFind(jsonResponse, "\"signals\":[]") >= 0) return;
   int signalsStart = StringFind(jsonResponse, "\"signals\":[");
   if(signalsStart < 0) return;
   
   int pos = signalsStart;
   while(true)
   {
      int objStart = StringFind(jsonResponse, "{\"id\":", pos);
      if(objStart < 0) break;
      int objEnd = StringFind(jsonResponse, "}", objStart);
      if(objEnd < 0) break;
      string signalJson = StringSubstr(jsonResponse, objStart, objEnd - objStart + 1);
      ExecuteSignal(signalJson);
      pos = objEnd + 1;
   }
}

void ExecuteSignal(string signalJson)
{
   string signalId = ExtractJsonString(signalJson, "id");
   if(signalId == lastSignalId) return;
   
   string symbol = ExtractJsonString(signalJson, "symbol");
   string action = ExtractJsonString(signalJson, "action");
   double volume = ExtractJsonDouble(signalJson, "volume");
   double sl = ExtractJsonDouble(signalJson, "sl");
   double tp = ExtractJsonDouble(signalJson, "tp");
   string comment = ExtractJsonString(signalJson, "comment");
   
   if(symbol == "" || action == "" || volume <= 0) { Print("Invalid signal"); return; }
   
   Print("Processing signal: ", signalId, " - ", symbol, " ", action, " ", volume, " lots");
   
   if(!ENABLE_TRADING) { SendSignalResult(signalId, false, 0, "Trading disabled"); lastSignalId = signalId; return; }
   if(!SymbolSelect(symbol, true)) { SendSignalResult(signalId, false, 0, "Symbol not found"); lastSignalId = signalId; return; }
   
   // Check for fresh quotes before trading
   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick))
   {
      SendSignalResult(signalId, false, 0, "Cannot get tick data");
      lastSignalId = signalId;
      return;
   }
   
   // Check if quotes are stale (more than 60 seconds old = likely market closed/holiday)
   datetime tickAge = TimeCurrent() - tick.time;
   if(tickAge > 60)
   {
      Print("Stale quotes detected for ", symbol, " - tick age: ", tickAge, " seconds. Market may be closed.");
      SendSignalResult(signalId, false, 0, StringFormat("Stale quotes (%d sec old) - market may be closed", tickAge));
      lastSignalId = signalId;
      return;
   }
   
   double ask = tick.ask;
   double bid = tick.bid;
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   long stopLevel = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   long freezeLevel = SymbolInfoInteger(symbol, SYMBOL_TRADE_FREEZE_LEVEL);
   double minDist = MathMax(stopLevel, freezeLevel) * point;
   
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);
   double origSL = sl, origTP = tp;
   
   bool success = false;
   ulong ticket = 0;
   string errorMsg = "";
   
   Print("Executing: ", action, " ", symbol, " vol=", volume, " sl=", sl, " tp=", tp, " bid=", bid, " ask=", ask, " tick_age=", tickAge, "s");
   
   // Use price=0 to let MT5 use current market price (more robust than passing explicit price)
   if(action == "buy")
   {
      if(sl > 0 && sl > bid - minDist) sl = NormalizeDouble(bid - minDist - 10*point, digits);
      if(tp > 0 && tp < ask + minDist) tp = NormalizeDouble(ask + minDist + 10*point, digits);
      
      // First try with SL/TP, using price=0 for market order
      success = trade.Buy(volume, symbol, 0, sl, tp, comment);
      if(success) { ticket = trade.ResultOrder(); }
      else if(trade.ResultRetcode() == 10016 || trade.ResultRetcode() == 10021)
      {
         // Invalid stops or off quotes - try without SL/TP first, then modify
         success = trade.Buy(volume, symbol, 0, 0, 0, comment);
         if(success) { ticket = trade.ResultOrder(); Sleep(100); trade.PositionModify(ticket, origSL, origTP); }
         else { errorMsg = StringFormat("Buy failed: %d", trade.ResultRetcode()); }
      }
      else { errorMsg = StringFormat("Buy failed: %d - %s", trade.ResultRetcode(), trade.ResultRetcodeDescription()); }
   }
   else if(action == "sell")
   {
      if(sl > 0 && sl < ask + minDist) sl = NormalizeDouble(ask + minDist + 10*point, digits);
      if(tp > 0 && tp > bid - minDist) tp = NormalizeDouble(bid - minDist - 10*point, digits);
      
      // First try with SL/TP, using price=0 for market order
      success = trade.Sell(volume, symbol, 0, sl, tp, comment);
      if(success) { ticket = trade.ResultOrder(); }
      else if(trade.ResultRetcode() == 10016 || trade.ResultRetcode() == 10021)
      {
         // Invalid stops or off quotes - try without SL/TP first, then modify
         success = trade.Sell(volume, symbol, 0, 0, 0, comment);
         if(success) { ticket = trade.ResultOrder(); Sleep(100); trade.PositionModify(ticket, origSL, origTP); }
         else { errorMsg = StringFormat("Sell failed: %d", trade.ResultRetcode()); }
      }
      else { errorMsg = StringFormat("Sell failed: %d - %s", trade.ResultRetcode(), trade.ResultRetcodeDescription()); }
   }
   else if(action == "close")
   {
      ulong posTicket = (ulong)ExtractJsonDouble(signalJson, "ticket");
      if(posTicket > 0) { success = trade.PositionClose(posTicket); if(!success) errorMsg = StringFormat("Close failed: %d", trade.ResultRetcode()); }
   }
   else if(action == "modify")
   {
      ulong posTicket = (ulong)ExtractJsonDouble(signalJson, "ticket");
      if(posTicket > 0) { success = trade.PositionModify(posTicket, sl, tp); if(!success) errorMsg = StringFormat("Modify failed: %d", trade.ResultRetcode()); }
   }
   
   SendSignalResult(signalId, success, ticket, errorMsg);
   lastSignalId = signalId;
   if(success) Print("Signal executed. Ticket: ", ticket);
   else Print("Signal failed: ", errorMsg);
}

void SendSignalResult(string signalId, bool success, ulong ticket, string error)
{
   string url = API_URL + "/api/signal_result";
   string headers = "Content-Type: application/json\r\n";
   string jsonBody = StringFormat("{\"signal_id\":\"%s\",\"success\":%s,\"ticket\":%d,\"error\":\"%s\"}", signalId, success ? "true" : "false", ticket, error);
   char data[], result[]; string resultHeaders;
   StringToCharArray(jsonBody, data, 0, StringLen(jsonBody));
   ArrayResize(data, StringLen(jsonBody));
   WebRequest("POST", url, headers, 5000, data, result, resultHeaders);
}

void SendPositionUpdates()
{
   string url = API_URL + "/api/positions";
   string headers = "Content-Type: application/json\r\n";
   string positionsJson = "[";
   int total = PositionsTotal();
   
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(i > 0) positionsJson += ",";
         positionsJson += StringFormat(
            "{\"ticket\":%d,\"symbol\":\"%s\",\"type\":\"%s\",\"volume\":%.2f,\"price_open\":%.5f,\"price_current\":%.5f,\"sl\":%.5f,\"tp\":%.5f,\"profit\":%.2f,\"swap\":%.2f,\"time\":%d}",
            ticket, PositionGetString(POSITION_SYMBOL),
            PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "buy" : "sell",
            PositionGetDouble(POSITION_VOLUME), PositionGetDouble(POSITION_PRICE_OPEN),
            PositionGetDouble(POSITION_PRICE_CURRENT), PositionGetDouble(POSITION_SL),
            PositionGetDouble(POSITION_TP), PositionGetDouble(POSITION_PROFIT),
            PositionGetDouble(POSITION_SWAP), PositionGetInteger(POSITION_TIME));
      }
   }
   positionsJson += "]";
   
   string jsonBody = "{\"positions\":" + positionsJson + "}";
   char data[], result[]; string resultHeaders;
   StringToCharArray(jsonBody, data, 0, StringLen(jsonBody));
   ArrayResize(data, StringLen(jsonBody));
   WebRequest("POST", url, headers, 5000, data, result, resultHeaders);
}

//+------------------------------------------------------------------+
//| Position Management Functions - Trailing, Break-Even, Partial   |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
   int total = PositionsTotal();
   
   // Clean up states for closed positions
   CleanupPositionStates();
   
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      
      // Only manage our positions
      long magic = PositionGetInteger(POSITION_MAGIC);
      if(magic != 0 && magic != MAGIC_NUMBER) continue;
      
      string symbol = PositionGetString(POSITION_SYMBOL);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      double currentSL = PositionGetDouble(POSITION_SL);
      double currentTP = PositionGetDouble(POSITION_TP);
      double volume = PositionGetDouble(POSITION_VOLUME);
      double profit = PositionGetDouble(POSITION_PROFIT);
      
      // Get symbol info
      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
      long stopLevel = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
      long freezeLevel = SymbolInfoInteger(symbol, SYMBOL_TRADE_FREEZE_LEVEL);
      double minDist = MathMax(stopLevel, freezeLevel) * point;
      double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
      double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
      
      // Get or create position state
      int stateIdx = GetPositionStateIndex(ticket);
      if(stateIdx < 0)
      {
         stateIdx = AddPositionState(ticket, entryPrice, currentSL, currentTP);
         if(stateIdx < 0) continue;
      }
      
      // Calculate profit in pips
      double profitPips = 0;
      if(posType == POSITION_TYPE_BUY)
         profitPips = (currentPrice - entryPrice) / point;
      else
         profitPips = (entryPrice - currentPrice) / point;
      
      // 1. Break-Even Logic
      if(ENABLE_BREAKEVEN && !positionStates[stateIdx].breakEvenApplied && profitPips >= BE_TRIGGER_PIPS * 10)
      {
         double newSL = 0;
         if(posType == POSITION_TYPE_BUY)
            newSL = NormalizeDouble(entryPrice + BE_OFFSET_PIPS * 10 * point, digits);
         else
            newSL = NormalizeDouble(entryPrice - BE_OFFSET_PIPS * 10 * point, digits);
         
         // Only modify if new SL is better than current
         bool shouldModify = false;
         if(posType == POSITION_TYPE_BUY && (currentSL == 0 || newSL > currentSL))
            shouldModify = true;
         else if(posType == POSITION_TYPE_SELL && (currentSL == 0 || newSL < currentSL))
            shouldModify = true;
         
         if(shouldModify && IsValidStopLevel(symbol, posType, newSL, currentPrice, minDist))
         {
            if(trade.PositionModify(ticket, newSL, currentTP))
            {
               positionStates[stateIdx].breakEvenApplied = true;
               Print("Break-even applied for ticket ", ticket, " - new SL: ", newSL);
            }
         }
      }
      
      // 2. Partial Close at 1:1 RR
      if(ENABLE_PARTIAL_CLOSE && !positionStates[stateIdx].partialClosed && currentTP > 0)
      {
         double riskPips = 0;
         if(posType == POSITION_TYPE_BUY)
            riskPips = (entryPrice - positionStates[stateIdx].originalSL) / point;
         else
            riskPips = (positionStates[stateIdx].originalSL - entryPrice) / point;
         
         // Check if we've reached 1:1 RR (profit >= risk)
         if(riskPips > 0 && profitPips >= riskPips)
         {
            double closeVolume = NormalizeVolume(volume * PARTIAL_CLOSE_PERCENT / 100.0, minLot, lotStep);
            double remainingVolume = volume - closeVolume;
            
            // Only partial close if both close and remaining volumes are valid
            if(closeVolume >= minLot && remainingVolume >= minLot)
            {
               if(trade.PositionClosePartial(ticket, closeVolume))
               {
                  positionStates[stateIdx].partialClosed = true;
                  Print("Partial close (", PARTIAL_CLOSE_PERCENT, "%) for ticket ", ticket, " - closed ", closeVolume, " lots");
               }
            }
            else if(closeVolume >= minLot && remainingVolume < minLot)
            {
               // Can't partial close without going below min lot - mark as done to avoid repeated attempts
               positionStates[stateIdx].partialClosed = true;
               Print("Partial close skipped for ticket ", ticket, " - remaining volume would be below min lot");
            }
         }
      }
      
      // 3. Trailing Stop
      if(ENABLE_TRAILING && profitPips >= TRAIL_START_PIPS * 10)
      {
         double newSL = 0;
         if(posType == POSITION_TYPE_BUY)
            newSL = NormalizeDouble(currentPrice - TRAIL_DISTANCE_PIPS * 10 * point, digits);
         else
            newSL = NormalizeDouble(currentPrice + TRAIL_DISTANCE_PIPS * 10 * point, digits);
         
         // Only trail if new SL is better than current (tighter)
         bool shouldTrail = false;
         double minImprovement = 5 * point;  // Minimum 0.5 pip improvement to avoid spam
         
         if(posType == POSITION_TYPE_BUY && (currentSL == 0 || newSL > currentSL + minImprovement))
            shouldTrail = true;
         else if(posType == POSITION_TYPE_SELL && (currentSL == 0 || newSL < currentSL - minImprovement))
            shouldTrail = true;
         
         if(shouldTrail && IsValidStopLevel(symbol, posType, newSL, currentPrice, minDist))
         {
            if(trade.PositionModify(ticket, newSL, currentTP))
               Print("Trailing stop updated for ticket ", ticket, " - new SL: ", newSL);
         }
      }
   }
}

int GetPositionStateIndex(ulong ticket)
{
   for(int i = 0; i < positionStateCount; i++)
   {
      if(positionStates[i].ticket == ticket)
         return i;
   }
   return -1;
}

int AddPositionState(ulong ticket, double entryPrice, double sl, double tp)
{
   if(positionStateCount >= ArraySize(positionStates))
   {
      ArrayResize(positionStates, positionStateCount + 10);
   }
   
   positionStates[positionStateCount].ticket = ticket;
   positionStates[positionStateCount].breakEvenApplied = false;
   positionStates[positionStateCount].partialClosed = false;
   positionStates[positionStateCount].originalSL = sl;
   positionStates[positionStateCount].originalTP = tp;
   positionStates[positionStateCount].entryPrice = entryPrice;
   
   positionStateCount++;
   return positionStateCount - 1;
}

void CleanupPositionStates()
{
   // Remove states for positions that no longer exist
   for(int i = positionStateCount - 1; i >= 0; i--)
   {
      bool found = false;
      for(int j = 0; j < PositionsTotal(); j++)
      {
         if(PositionGetTicket(j) == positionStates[i].ticket)
         {
            found = true;
            break;
         }
      }
      
      if(!found)
      {
         // Remove this state by shifting remaining elements
         for(int k = i; k < positionStateCount - 1; k++)
         {
            positionStates[k] = positionStates[k + 1];
         }
         positionStateCount--;
      }
   }
}

bool IsValidStopLevel(string symbol, ENUM_POSITION_TYPE posType, double sl, double currentPrice, double minDist)
{
   if(sl <= 0) return false;
   
   double distance = 0;
   if(posType == POSITION_TYPE_BUY)
      distance = currentPrice - sl;
   else
      distance = sl - currentPrice;
   
   return distance >= minDist;
}

double NormalizeVolume(double volume, double minLot, double lotStep)
{
   // Round down to nearest lot step
   double normalized = MathFloor(volume / lotStep) * lotStep;
   if(normalized < minLot) return 0;
   return normalized;
}

string ExtractJsonString(string json, string key)
{
   string searchKey = "\"" + key + "\":\"";
   int start = StringFind(json, searchKey);
   if(start < 0) return "";
   start += StringLen(searchKey);
   int end = StringFind(json, "\"", start);
   if(end < 0) return "";
   return StringSubstr(json, start, end - start);
}

double ExtractJsonDouble(string json, string key)
{
   string searchKey = "\"" + key + "\":";
   int start = StringFind(json, searchKey);
   if(start < 0) return 0;
   start += StringLen(searchKey);
   int end = start;
   while(end < StringLen(json))
   {
      ushort c = StringGetCharacter(json, end);
      if(c == ',' || c == '}' || c == ' ') break;
      end++;
   }
   return StringToDouble(StringSubstr(json, start, end - start));
}
