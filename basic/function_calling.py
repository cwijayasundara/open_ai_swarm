from swarm import Swarm, Agent
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
_ = load_dotenv()

client = Swarm()

def get_weather() -> str:
    return "{'temp':67, 'unit':'F'}"

agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
    functions=[get_weather],
)

messages = [{"role": "user", "content": "What's the weather in NYC?"}]

response = client.run(agent=agent, messages=messages)
print(response.messages[-1]["content"])