#!/usr/bin/env python3
"""Stock portfolio agent powered by Claude with tool use."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic

PORTFOLIO_FILE = Path("portfolio.json")
MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are a helpful stock portfolio assistant. You help users:
- Track their stock investments (add/remove positions)
- Check current stock prices and detailed company info
- Analyze portfolio performance (total value, P&L, allocation)
- Provide concise market context on individual stocks

Guidelines:
- Format currency as $X,XXX.XX with 2 decimal places
- Show P&L with + for gains, - for losses, and the % change
- Be concise but informative; interpret the numbers for the user
- If a price lookup fails, mention it clearly and work with available data"""


# ── Portfolio persistence ──────────────────────────────────────────────────────

def _load() -> dict:
    if PORTFOLIO_FILE.exists():
        return json.loads(PORTFOLIO_FILE.read_text())
    return {"positions": {}, "transactions": []}


def _save(portfolio: dict) -> None:
    PORTFOLIO_FILE.write_text(json.dumps(portfolio, indent=2))


# ── Tool implementations ───────────────────────────────────────────────────────

def get_stock_price(symbol: str) -> dict:
    """Fetch the current market price for a ticker symbol."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        if price is None:
            return {"error": f"Could not retrieve price for {symbol.upper()}"}
        return {
            "symbol": symbol.upper(),
            "price": round(float(price), 2),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
        }
    except ImportError:
        return {"error": "yfinance not installed. Run: pip install yfinance"}
    except Exception as e:
        return {"error": str(e)}


def get_stock_info(symbol: str) -> dict:
    """Fetch detailed stock information: price, fundamentals, sector, description."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        return {
            "symbol": symbol.upper(),
            "name": info.get("longName", symbol.upper()),
            "price": round(float(price), 2) if price else None,
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "dividend_yield": info.get("dividendYield"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "description": (info.get("longBusinessSummary") or "")[:600] or None,
        }
    except ImportError:
        return {"error": "yfinance not installed. Run: pip install yfinance"}
    except Exception as e:
        return {"error": str(e)}


def add_position(symbol: str, shares: float, purchase_price: float) -> dict:
    """Add shares to the portfolio. Averages cost basis if position already exists."""
    portfolio = _load()
    symbol = symbol.upper()

    if symbol in portfolio["positions"]:
        existing = portfolio["positions"][symbol]
        total = existing["shares"] + shares
        avg = (existing["shares"] * existing["avg_purchase_price"] + shares * purchase_price) / total
        portfolio["positions"][symbol] = {"shares": round(total, 6), "avg_purchase_price": round(avg, 4)}
        action = "updated"
    else:
        portfolio["positions"][symbol] = {"shares": shares, "avg_purchase_price": purchase_price}
        action = "added"

    portfolio["transactions"].append({
        "date": datetime.now().isoformat(),
        "type": "buy",
        "symbol": symbol,
        "shares": shares,
        "price": purchase_price,
    })
    _save(portfolio)
    return {
        "success": True,
        "action": action,
        "symbol": symbol,
        "shares_added": shares,
        "purchase_price": purchase_price,
        "position": portfolio["positions"][symbol],
    }


def remove_position(symbol: str, shares: float | None = None) -> dict:
    """Remove shares or an entire position from the portfolio."""
    portfolio = _load()
    symbol = symbol.upper()

    if symbol not in portfolio["positions"]:
        return {"error": f"{symbol} not found in portfolio"}

    existing = portfolio["positions"][symbol]
    if shares is None or shares >= existing["shares"]:
        removed = existing["shares"]
        del portfolio["positions"][symbol]
        action = f"Removed entire position ({removed} shares)"
    else:
        portfolio["positions"][symbol]["shares"] = round(existing["shares"] - shares, 6)
        removed = shares
        action = f"Sold {shares} shares; {portfolio['positions'][symbol]['shares']} remaining"

    portfolio["transactions"].append({
        "date": datetime.now().isoformat(),
        "type": "sell",
        "symbol": symbol,
        "shares": removed,
    })
    _save(portfolio)
    return {"success": True, "symbol": symbol, "action": action}


def get_portfolio() -> dict:
    """Return all current positions with share counts and average purchase prices."""
    portfolio = _load()
    return {
        "positions": portfolio["positions"],
        "total_positions": len(portfolio["positions"]),
    }


def get_portfolio_value() -> dict:
    """Calculate total portfolio market value, cost basis, and P&L for all positions."""
    portfolio = _load()
    if not portfolio["positions"]:
        return {
            "total_value": 0,
            "total_cost": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
            "positions": [],
            "message": "Portfolio is empty.",
        }

    details = []
    total_value = 0.0
    total_cost = 0.0

    for symbol, pos in portfolio["positions"].items():
        price_data = get_stock_price(symbol)
        if "error" in price_data:
            current_price = pos["avg_purchase_price"]
            price_note = "using purchase price (live data unavailable)"
        else:
            current_price = price_data["price"]
            price_note = None

        market_value = current_price * pos["shares"]
        cost_basis = pos["avg_purchase_price"] * pos["shares"]
        pnl = market_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0

        total_value += market_value
        total_cost += cost_basis

        entry = {
            "symbol": symbol,
            "shares": pos["shares"],
            "avg_purchase_price": pos["avg_purchase_price"],
            "current_price": round(current_price, 2),
            "market_value": round(market_value, 2),
            "cost_basis": round(cost_basis, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
        }
        if price_note:
            entry["note"] = price_note
        details.append(entry)

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

    return {
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "positions": details,
    }


# ── Tool registry ──────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Get the current market price of a stock by its ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL, MSFT, TSLA"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_stock_info",
        "description": (
            "Get detailed information about a stock: current price, P/E ratio, "
            "52-week high/low, market cap, dividend yield, sector, industry, and "
            "a brief business description."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "add_position",
        "description": (
            "Add shares of a stock to the portfolio at a given purchase price. "
            "If the position already exists, shares are added and cost basis is averaged."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol"},
                "shares": {"type": "number", "description": "Number of shares to add"},
                "purchase_price": {"type": "number", "description": "Price per share at time of purchase"},
            },
            "required": ["symbol", "shares", "purchase_price"],
        },
    },
    {
        "name": "remove_position",
        "description": (
            "Remove shares or an entire position from the portfolio. "
            "If 'shares' is omitted, the entire position is removed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol"},
                "shares": {"type": "number", "description": "Shares to remove; omit to remove the full position"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_portfolio",
        "description": "List all positions in the portfolio with share counts and average purchase prices.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_portfolio_value",
        "description": (
            "Calculate the current total portfolio value, cost basis, and profit/loss "
            "for every position using live market prices."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

_TOOL_FN = {
    "get_stock_price":    lambda a: get_stock_price(a["symbol"]),
    "get_stock_info":     lambda a: get_stock_info(a["symbol"]),
    "add_position":       lambda a: add_position(a["symbol"], a["shares"], a["purchase_price"]),
    "remove_position":    lambda a: remove_position(a["symbol"], a.get("shares")),
    "get_portfolio":      lambda _: get_portfolio(),
    "get_portfolio_value": lambda _: get_portfolio_value(),
}


def _run_tool(name: str, tool_input: dict) -> str:
    fn = _TOOL_FN.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return json.dumps(fn(tool_input))
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    messages: list[dict] = []

    print("=" * 55)
    print("  Stock Portfolio Agent  📈")
    print("=" * 55)
    print("Try asking:")
    print('  • "Add 10 shares of AAPL at $180"')
    print('  • "What\'s my portfolio worth?"')
    print('  • "Tell me about NVDA"')
    print('  • "What\'s the price of TSLA?"')
    print('  • "Remove my MSFT position"')
    print('  • "Show me my portfolio"')
    print('Type "quit" to exit.\n')

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Agentic tool-use loop
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        print(f"\nAssistant: {block.text}\n")
                messages.append({"role": "assistant", "content": response.content})
                break

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  [tool: {block.name}]", flush=True)
                        result = _run_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})
            else:
                # Unexpected stop reason — surface the text if any
                for block in response.content:
                    if hasattr(block, "text"):
                        print(f"\nAssistant: {block.text}\n")
                messages.append({"role": "assistant", "content": response.content})
                break


if __name__ == "__main__":
    run()
