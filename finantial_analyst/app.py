from dotenv import load_dotenv
import requests
import os

_ = load_dotenv()

FINANCIAL_MODELING_PREP_API_KEY = os.getenv("FINANCIAL_MODELING_PREP_API_KEY")

def get_stock_price(symbol):
    """
    Fetch the current stock price for the given symbol, the current volume, the average price 50d and 200d, EPS, PE and
    the next earnings Announcement.
    """
    url = f"https://financialmodelingprep.com/api/v3/quote-order/{symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    try:
        price = data[0]['price']
        volume = data[0]['volume']
        priceAvg50 = data[0]['priceAvg50']
        priceAvg200 = data[0]['priceAvg200']
        eps = data[0]['eps']
        pe = data[0]['pe']
        earningsAnnouncement = data[0]['earningsAnnouncement']
        return {"symbol": symbol.upper(), "price": price, "volume":volume,"priceAvg50":priceAvg50, "priceAvg200":priceAvg200, "EPS":eps, "PE":pe, "earningsAnnouncement":earningsAnnouncement }
    except (IndexError, KeyError):
        return {"error": f"Could not fetch price for symbol: {symbol}"}

def get_company_financials(symbol):
    """
    Fetch basic financial information for the given company symbol such as the industry, the sector, the name of the
    company, and the market capitalization.
    """
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    try:
        results = data[0]
        financials = {
            "symbol": results["symbol"],
            "companyName": results["companyName"],
            "marketCap": results["mktCap"],
            "industry": results["industry"],
            "sector": results["sector"],
            "website": results["website"],
            "beta":results["beta"],
            "price":results["price"],
        }
        return financials
    except (IndexError, KeyError):
        return {"error": f"Could not fetch financials for symbol: {symbol}"}

def get_income_statement(symbol):
    """
    Fetch last income statement for the given company symbol such as revenue, gross profit, net income, EBITDA, EPS.
    """
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=annual&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    try:
        results = data[0]
        financials = {
            "date": results["date"],
            "revenue": results["revenue"],
            "gross profit": results["grossProfit"],
            "net Income": results["netIncome"],
            "ebitda": results["ebitda"],
            "EPS": results["eps"],
            "EPS diluted":results["epsdiluted"]
        }
        return data, financials
    except (IndexError, KeyError):
        return {"error": f"Could not fetch financials for symbol: {symbol}"}

from swarm import Agent

manager_agent = Agent(
    name="Manager Agent",
    instructions="Determine which agent is best suited to handle the user's request, and transfer the conversation to "
                 "that agent.",
)
stock_price_agent = Agent(
    name="Stock Price Agent",
    instructions="Fetch Historical prices for a given stock symbol, the current volume, the average price 50d and 200d, "
                 "EPS, PE and the next earnings Announcement.",
    functions=[get_stock_price],
)

company_basic_information_agent = Agent(
    name="Company basic information Agent",
    instructions="Fetch basic financial information for the given company symbol such as the industry, the sector, the "
                 "name of the company, and the market capitalization.",
    functions=[get_company_financials],
)

income_statement_agent = Agent(
    name="Income Statement Agent",
    instructions=" Fetch last income statement for the given company symbol such as revenue, gross profit, net income, "
                 "EBITDA, EPS.",
    functions=[get_income_statement],
)

def transfer_back_to_manager():
    """Call this function if a user is asking about a topic that is not handled by the current agent."""
    return manager_agent


def transfer_to_stock_price():
    return stock_price_agent

def transfer_to_company_basic_info():
    return company_basic_information_agent

def transfer_to_income_statement():
    return income_statement_agent


manager_agent.functions = [transfer_to_stock_price, transfer_to_company_basic_info, transfer_to_income_statement]
stock_price_agent.functions.append(transfer_back_to_manager)
company_basic_information_agent.functions.append(transfer_back_to_manager)
income_statement_agent.functions.append(transfer_back_to_manager)

# You can get this code from "from swarm.repl import run_demo_loop"==>I've extracted here to add "exit" to break the loop.
import json

from swarm import Swarm

def process_and_print_streaming_response(response):
    content = ""
    last_sender = ""

    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            if not content and last_sender:
                print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                last_sender = ""
            print(chunk["content"], end="", flush=True)
            content += chunk["content"]

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f["name"]
                if not name:
                    continue
                print(f"\033[94m{last_sender}: \033[95m{name}\033[0m()")

        if "delim" in chunk and chunk["delim"] == "end" and content:
            print()  # End of response message
            content = ""

        if "response" in chunk:
            return chunk["response"]

def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")

def run_demo_loop(
    starting_agent, context_variables=None, stream=False, debug=False
) -> None:
    client = Swarm()
    print("Starting Swarm CLI 🐝")

    messages = []
    agent = starting_agent

    while True:
        user_input = input("\033[90mUser\033[0m: ")
        if user_input.lower() == "exit":
            break
        messages.append({"role": "user", "content": user_input})

        response = client.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
            stream=stream,
            debug=debug,
        )

        if stream:
            response = process_and_print_streaming_response(response)
        else:
            pretty_print_messages(response.messages)

        messages.extend(response.messages)
        agent = response.agent

run_demo_loop(manager_agent, debug=False)
