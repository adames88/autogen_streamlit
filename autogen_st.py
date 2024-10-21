import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import autogen
from utils import get_openai_api_key
from autogen.coding import LocalCommandLineCodeExecutor

# Set up the OpenAI API key
get_openai_api_key()

llm_config = {"model": "gpt-4-turbo"}

# Avatars for each agent (using emojis)
avatars = {
    "Admin": "👨‍💼",
    "Planner": "🗓",
    "Engineer": "👩‍💻",
    "Executor": "🛠",
    "Writer": "✍"
}

# Custom stock data retrieval and plotting functions
def get_stock_prices(stock_symbols, start_date, end_date):
    stock_data = yf.download(stock_symbols, start=start_date, end=end_date)
    return stock_data.get("Close")

def plot_stock_prices(stock_prices, filename):
    plt.figure(figsize=(10, 5))
    for column in stock_prices.columns:
        plt.plot(stock_prices.index, stock_prices[column], label=column)
    plt.title("Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Define Executor with provided functions
executor_func = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
    functions=[get_stock_prices, plot_stock_prices],
)

executor = autogen.ConversableAgent(
    name="Executor",
    description="Execute the code written by the Engineer and report the result. Execute multiple steps if provided."
                "When you have fully completed the execution of code successfully give the results and information to the planner to prepare for the writer."
                "Save graph, visualisation plots and data in the current directory.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 5, "executor": executor_func},
)

# Planner with enhanced multi-step handling
planner = autogen.ConversableAgent(
    name="Planner",
    system_message=(
        "You are responsible for planning the task. Break it down into steps and coordinate with Engineer for code "
        "and Executor for execution. If steps fail, guide the agents to retry. Never ask the engineer to run code, only provide plan for the engineer to write the code."
    ),
    description="Plan and delegate tasks in a step-by-step manner and ensure successful task completion.",
    llm_config=llm_config,
)

# Engineer to write code based on Planner instructions
engineer = autogen.AssistantAgent(
    name="Engineer",
    system_message=(
        "Write code based on the Planner's instructions. Communicate with the Executor to run the code. "
        "Never run code, only write code and pass it to the executor to run."
    ),
    description="Write and iterate code for stock price retrieval and analysis based on Planner's instructions.",
    llm_config=llm_config,
)

# Writer Agent
writer = autogen.ConversableAgent(
    name="Writer",
    system_message="Write the final report after analysis. Refine based on feedback."
                   "When you have written your final report save it in current directory as a markdown file."
                   "Present the information in a clean, professional, user-friendly and aesthetically pleasing presentation.",
    description="Write and refine reports based on the results of the analysis.",
    llm_config=llm_config,
)

# Define Admin Agent (user_proxy)
user_proxy = autogen.ConversableAgent(
    name="Admin",
    system_message="Oversee the workflow. Ensure Planner creates a step-by-step plan and delegates tasks correctly.",
    code_execution_config=False,
    llm_config=llm_config,
    human_input_mode="ALWAYS",
)

# Group Chat for Agents
groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, writer, executor, planner],
    messages=[],
    max_round=50,
    allowed_or_disallowed_speaker_transitions={
        user_proxy: [planner],
        planner: [engineer, executor, writer],
        engineer: [executor],
        executor: [planner],
        writer: [user_proxy, planner],
    },
    speaker_transitions_type="allowed",
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Streamlit UI Setup
st.title("Agent Conversation and Task Management")

# Task input
task_input = st.text_input("Enter your task (e.g., Retrieve stock prices for analysis):")

# Function to display messages in Streamlit with avatars
def print_messages(recipient, messages, sender, config):
    content = messages[-1]['content']
    user_name = messages[-1].get('name', sender.name)
    user_avatar = avatars.get(user_name, "")
    st.write(f"{user_avatar} **{user_name}:** {content}")
    return False, None

# Register reply functions to capture and display messages with avatars
user_proxy.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
engineer.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
planner.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
executor.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
writer.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)

# Function to initiate the workflow
if st.button("Submit Task"):
    if task_input:
        st.write(f"**Task:** {task_input}")
        # Start the chat between Admin and Planner
        groupchat_result = user_proxy.initiate_chat(
            manager, message=f"Admin initiated the task: {task_input}"
        )
        st.write(f"**Chat Manager Result:** {groupchat_result}")

# Placeholder for results
st.write("### Results")
