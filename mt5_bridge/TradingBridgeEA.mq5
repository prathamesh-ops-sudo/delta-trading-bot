//+------------------------------------------------------------------+
//|                                            TradingBridgeEA.mq5   |
//|                                    AI Trading System Bridge      |
//|                                  Connects to AWS Trading Brain   |
//+------------------------------------------------------------------+
#property copyright "AI Trading System"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

input string   API_URL = "http://13.222.99.140:5000";  // AWS API URL
input int      POLL_INTERVAL = 5;                       // Poll interval in seconds
input int      MAGIC_NUMBER = 123456;                   // Magic number for trades
input double   MAX_SLIPPAGE = 20;                       // Maximum slippage in points
input bool     ENABLE_TRADING = true;                   // Enable live trading

CTrade trade;
int lastPollTime = 0;
string lastSignalId = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MAGIC_NUMBER);
   trade.SetDeviationInPoints((ulong)MAX_SLIPPAGE);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   Print("TradingBridgeEA initialized");
   Print("API URL: ", API_URL);
   Print("Poll Interval: ", POLL_INTERVAL, " seconds");
   Print("Trading Enabled: ", ENABLE_TRADING);
   
   // Register with the API
   RegisterWithAPI();
   
   // Set timer for polling
   EventSetTimer(POLL_INTERVAL);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("TradingBridgeEA stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Timer function - polls API for signals                            |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Send account info to API
   SendAccountInfo();
   
   // Poll for trade signals
   PollForSignals();
   
   // Send position updates
   SendPositionUpdates();
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   // Can add tick-based logic here if needed
}

//+------------------------------------------------------------------+
//| Register EA with the API                                          |
//+------------------------------------------------------------------+
void RegisterWithAPI()
{
   string url = API_URL + "/api/register";
   string headers = "Content-Type: application/json\r\n";
   
   string jsonBody = StringFormat(
      "{\"login\":%d,\"server\":\"%s\",\"balance\":%.2f,\"equity\":%.2f,\"leverage\":%d}",
      AccountInfoInteger(ACCOUNT_LOGIN),
      AccountInfoString(ACCOUNT_SERVER),
      AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoDouble(ACCOUNT_EQUITY),
      AccountInfoInteger(ACCOUNT_LEVERAGE)
   );
   
   char data[];
   char result[];
   string resultHeaders;
   
   StringToCharArray(jsonBody, data, 0, StringLen(jsonBody));
   ArrayResize(data, StringLen(jsonBody));
   
   int res = WebRequest("POST", url, headers, 5000, data, result, resultHeaders);
   
   if(res == -1)
   {
      int error = GetLastError();
      Print("Registration failed. Error: ", error);
      if(error == 4060)
         Print("Add ", API_URL, " to allowed URLs in Tools -> Options -> Expert Advisors");
   }
   else
   {
      string response = CharArrayToString(result);
      Print("Registered with API: ", response);
   }
}

//+------------------------------------------------------------------+
//| Send account information to API                                   |
//+------------------------------------------------------------------+
void SendAccountInfo()
{
   string url = API_URL + "/api/account";
   string headers = "Content-Type: application/json\r\n";
   
   string jsonBody = StringFormat(
      "{\"login\":%d,\"server\":\"%s\",\"balance\":%.2f,\"equity\":%.2f,\"margin\":%.2f,\"free_margin\":%.2f,\"margin_level\":%.2f,\"leverage\":%d,\"profit\":%.2f,\"currency\":\"%s\"}",
      AccountInfoInteger(ACCOUNT_LOGIN),
      AccountInfoString(ACCOUNT_SERVER),
      AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoDouble(ACCOUNT_EQUITY),
      AccountInfoDouble(ACCOUNT_MARGIN),
      AccountInfoDouble(ACCOUNT_MARGIN_FREE),
      AccountInfoDouble(ACCOUNT_MARGIN_LEVEL),
      AccountInfoInteger(ACCOUNT_LEVERAGE),
      AccountInfoDouble(ACCOUNT_PROFIT),
      AccountInfoString(ACCOUNT_CURRENCY)
   );
   
   char data[];
   char result[];
   string resultHeaders;
   
   StringToCharArray(jsonBody, data, 0, StringLen(jsonBody));
   ArrayResize(data, StringLen(jsonBody));
   
   int res = WebRequest("POST", url, headers, 5000, data, result, resultHeaders);
   
   if(res == -1)
   {
      int error = GetLastError();
      if(error != 4060)  // Don't spam URL error
         Print("Account update failed. Error: ", error);
   }
}

//+------------------------------------------------------------------+
//| Poll API for trade signals                                        |
//+------------------------------------------------------------------+
void PollForSignals()
{
   string url = API_URL + "/api/signals";
   string headers = "Content-Type: application/json\r\n";
   
   char data[];
   char result[];
   string resultHeaders;
   
   int res = WebRequest("GET", url, headers, 5000, data, result, resultHeaders);
   
   if(res == -1)
   {
      int error = GetLastError();
      if(error != 4060)
         Print("Signal poll failed. Error: ", error);
      return;
   }
   
   string response = CharArrayToString(result);
   
   // Parse and execute signals
   ProcessSignals(response);
}

//+------------------------------------------------------------------+
//| Process trade signals from API                                    |
//+------------------------------------------------------------------+
void ProcessSignals(string jsonResponse)
{
   // Simple JSON parsing for signals array
   // Format: {"signals":[{"id":"xxx","symbol":"EURUSD","action":"buy","volume":0.01,"sl":1.0900,"tp":1.1100}]}
   
   if(StringFind(jsonResponse, "\"signals\":[]") >= 0)
      return;  // No signals
   
   int signalsStart = StringFind(jsonResponse, "\"signals\":[");
   if(signalsStart < 0)
      return;
   
   // Find each signal object
   int pos = signalsStart;
   while(true)
   {
      int objStart = StringFind(jsonResponse, "{\"id\":", pos);
      if(objStart < 0)
         break;
      
      int objEnd = StringFind(jsonResponse, "}", objStart);
      if(objEnd < 0)
         break;
      
      string signalJson = StringSubstr(jsonResponse, objStart, objEnd - objStart + 1);
      ExecuteSignal(signalJson);
      
      pos = objEnd + 1;
   }
}

//+------------------------------------------------------------------+
//| Execute a single trade signal                                     |
//+------------------------------------------------------------------+
void ExecuteSignal(string signalJson)
{
   // Parse signal fields
   string signalId = ExtractJsonString(signalJson, "id");
   
   // Skip if already processed
   if(signalId == lastSignalId)
      return;
   
   string symbol = ExtractJsonString(signalJson, "symbol");
   string action = ExtractJsonString(signalJson, "action");
   double volume = ExtractJsonDouble(signalJson, "volume");
   double sl = ExtractJsonDouble(signalJson, "sl");
   double tp = ExtractJsonDouble(signalJson, "tp");
   string comment = ExtractJsonString(signalJson, "comment");
   
   if(symbol == "" || action == "" || volume <= 0)
   {
      Print("Invalid signal: ", signalJson);
      return;
   }
   
   Print("Processing signal: ", signalId, " - ", symbol, " ", action, " ", volume, " lots");
   
   if(!ENABLE_TRADING)
   {
      Print("Trading disabled - signal not executed");
      SendSignalResult(signalId, false, 0, "Trading disabled");
      lastSignalId = signalId;
      return;
   }
   
   // Ensure symbol is selected
   if(!SymbolSelect(symbol, true))
   {
      Print("Failed to select symbol: ", symbol);
      SendSignalResult(signalId, false, 0, "Symbol not found");
      lastSignalId = signalId;
      return;
   }
   
   // Get current price and symbol info
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   long stopLevel = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minStopDistance = stopLevel * point;
   
   // Normalize SL/TP to symbol digits
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);
   
   // Validate and adjust SL/TP if too close to price
   if(action == "buy" && sl > 0)
   {
      double minSL = NormalizeDouble(ask - minStopDistance, digits);
      if(sl > minSL) sl = minSL;
   }
   else if(action == "sell" && sl > 0)
   {
      double minSL = NormalizeDouble(bid + minStopDistance, digits);
      if(sl < minSL) sl = minSL;
   }
   
   bool success = false;
   ulong ticket = 0;
   string errorMsg = "";
   
   Print("Executing: ", action, " ", symbol, " vol=", volume, " sl=", sl, " tp=", tp, " stopLevel=", stopLevel);
   
   if(action == "buy")
   {
      success = trade.Buy(volume, symbol, ask, sl, tp, comment);
      if(success)
         ticket = trade.ResultOrder();
      else
         errorMsg = StringFormat("Buy failed: %d - %s", trade.ResultRetcode(), trade.ResultRetcodeDescription());
   }
   else if(action == "sell")
   {
      success = trade.Sell(volume, symbol, bid, sl, tp, comment);
      if(success)
         ticket = trade.ResultOrder();
      else
         errorMsg = StringFormat("Sell failed: %d - %s", trade.ResultRetcode(), trade.ResultRetcodeDescription());
   }
   else if(action == "close")
   {
      ulong posTicket = (ulong)ExtractJsonDouble(signalJson, "ticket");
      if(posTicket > 0)
      {
         success = trade.PositionClose(posTicket);
         if(!success)
            errorMsg = StringFormat("Close failed: %d", trade.ResultRetcode());
      }
   }
   else if(action == "modify")
   {
      ulong posTicket = (ulong)ExtractJsonDouble(signalJson, "ticket");
      if(posTicket > 0)
      {
         success = trade.PositionModify(posTicket, sl, tp);
         if(!success)
            errorMsg = StringFormat("Modify failed: %d", trade.ResultRetcode());
      }
   }
   
   // Send result back to API
   SendSignalResult(signalId, success, ticket, errorMsg);
   
   lastSignalId = signalId;
   
   if(success)
      Print("Signal executed successfully. Ticket: ", ticket);
   else
      Print("Signal execution failed: ", errorMsg);
}

//+------------------------------------------------------------------+
//| Send signal execution result to API                               |
//+------------------------------------------------------------------+
void SendSignalResult(string signalId, bool success, ulong ticket, string error)
{
   string url = API_URL + "/api/signal_result";
   string headers = "Content-Type: application/json\r\n";
   
   string jsonBody = StringFormat(
      "{\"signal_id\":\"%s\",\"success\":%s,\"ticket\":%d,\"error\":\"%s\"}",
      signalId,
      success ? "true" : "false",
      ticket,
      error
   );
   
   char data[];
   char result[];
   string resultHeaders;
   
   StringToCharArray(jsonBody, data, 0, StringLen(jsonBody));
   ArrayResize(data, StringLen(jsonBody));
   
   WebRequest("POST", url, headers, 5000, data, result, resultHeaders);
}

//+------------------------------------------------------------------+
//| Send position updates to API                                      |
//+------------------------------------------------------------------+
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
         if(i > 0)
            positionsJson += ",";
         
         positionsJson += StringFormat(
            "{\"ticket\":%d,\"symbol\":\"%s\",\"type\":\"%s\",\"volume\":%.2f,\"price_open\":%.5f,\"price_current\":%.5f,\"sl\":%.5f,\"tp\":%.5f,\"profit\":%.2f,\"swap\":%.2f,\"time\":%d}",
            ticket,
            PositionGetString(POSITION_SYMBOL),
            PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "buy" : "sell",
            PositionGetDouble(POSITION_VOLUME),
            PositionGetDouble(POSITION_PRICE_OPEN),
            PositionGetDouble(POSITION_PRICE_CURRENT),
            PositionGetDouble(POSITION_SL),
            PositionGetDouble(POSITION_TP),
            PositionGetDouble(POSITION_PROFIT),
            PositionGetDouble(POSITION_SWAP),
            PositionGetInteger(POSITION_TIME)
         );
      }
   }
   
   positionsJson += "]";
   
   string jsonBody = "{\"positions\":" + positionsJson + "}";
   
   char data[];
   char result[];
   string resultHeaders;
   
   StringToCharArray(jsonBody, data, 0, StringLen(jsonBody));
   ArrayResize(data, StringLen(jsonBody));
   
   WebRequest("POST", url, headers, 5000, data, result, resultHeaders);
}

//+------------------------------------------------------------------+
//| Extract string value from JSON                                    |
//+------------------------------------------------------------------+
string ExtractJsonString(string json, string key)
{
   string searchKey = "\"" + key + "\":\"";
   int start = StringFind(json, searchKey);
   if(start < 0)
      return "";
   
   start += StringLen(searchKey);
   int end = StringFind(json, "\"", start);
   if(end < 0)
      return "";
   
   return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Extract double value from JSON                                    |
//+------------------------------------------------------------------+
double ExtractJsonDouble(string json, string key)
{
   string searchKey = "\"" + key + "\":";
   int start = StringFind(json, searchKey);
   if(start < 0)
      return 0;
   
   start += StringLen(searchKey);
   
   // Find end of number (comma, }, or end of string)
   int end = start;
   while(end < StringLen(json))
   {
      ushort c = StringGetCharacter(json, end);
      if(c == ',' || c == '}' || c == ' ')
         break;
      end++;
   }
   
   string valueStr = StringSubstr(json, start, end - start);
   return StringToDouble(valueStr);
}
//+------------------------------------------------------------------+
