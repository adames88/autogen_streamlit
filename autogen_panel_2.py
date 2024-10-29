import autogen
import panel as pn
import yfinance as yf
import matplotlib.pyplot as plt
from tool.utils import get_openai_api_key
from autogen.coding import LocalCommandLineCodeExecutor

get_openai_api_key()

llm_config = {"model": "gpt-4-turbo"}

# Custom stock data retrieval and plotting functions
def get_stock_prices(stock_symbols, start_date, end_date):
    """Get the stock prices for the given stock symbols between the start and end dates."""
    stock_data = yf.download(stock_symbols, start=start_date, end=end_date)
    return stock_data.get("Close")

def plot_stock_prices(stock_prices, filename):
    """Plot the stock prices for the given stock symbols."""
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
    "when you have fully completed the execution of code successfully give the results and information to the planner to prepare for the writer."
    "save graph, visualisation plots and data in the current directory.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 5,
        "executor": executor_func,
    },
)

# Planner with enhanced multi-step handling
planner = autogen.ConversableAgent(
    name="Planner",
    system_message=(
        "You are responsible for planning the task. Break it down into steps and coordinate with Engineer for code "
        "and Executor for execution. If steps fail, guide the agents to retry."
        "Never ask the engineer to run code, only provide plan for the engineer to write the code."
        "code should only be executed by the executor."
        "when sufficient information or data has been retrieved, send the results to writer with a plan for writing the report."
        "overlook the report written by the writer, if feedback is needed then provide, if not TERMINATE session."
    ),
    description="Plan and delegate tasks in a step-by-step manner and ensure successful task completion.",
    llm_config=llm_config,
)

# Engineer to write code based on Planner instructions
engineer = autogen.AssistantAgent(
    name="Engineer",
    system_message=(
        "Write code based on the Planner's instructions. Communicate with the Executor to run the code. "
        "If a task fails, modify the code and reattempt execution."
        "never run code, only write code and pass it to the executor to run."
        "when writing code dont include executable functions, let the executor run the functions."
        "write a python runnable script which the executor can run"
        "when creating python script save it in coding folder, do not run it."
    ),
    description="Write and iterate code for stock price retrieval and analysis based on Planner's instructions.",
    llm_config=llm_config,
)

# Writer Agent
writer = autogen.ConversableAgent(
    name="Writer",
    system_message="Write the final report after analysis. Refine based on feedback."
    "when you have written your final report save it in current directory as a markdown file."
    "present the information in a clean, professional, user-friendly and aesthetically pleasing presentation.",
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

# Avatars for each agent (using emojis)
avatars = {
    user_proxy.name: "👨‍💼",  # Admin
    planner.name: "🗓",  # Planner
    engineer.name: "👩‍💻",  # Engineer
    executor.name: "🛠",  # Executor
    writer.name: "✍"  # Writer
}

# Group Chat for Agents
groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, writer, executor, planner],
    messages=[],
    max_round=50,
    allowed_or_disallowed_speaker_transitions={
        user_proxy: [planner],  # Admin delegates to Planner first
        planner: [engineer, executor, writer],  # Planner can delegate to Engineer, Executor, or Writer
        engineer: [executor],  # Engineer delegates execution to Executor
        executor: [planner],  # Executor reports back to Planner or sends results to Writer
        writer: [user_proxy, planner],  # Writer can ask for feedback from Admin or Planner
    },
    speaker_transitions_type="allowed",
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# UI Setup with Panel
pn.extension(design="material")

# Task Input
task_input = pn.widgets.TextInput(name="Enter your task", placeholder="E.g., Retrieve stock prices for analysis.")
submit_button = pn.widgets.Button(name="Submit Task", button_type="primary")

# Chat Interface and Messages Display
chat_interface = pn.chat.ChatInterface()
chat_interface.send("Send a message!", user="System", respond=False)

# Function to display messages in Panel UI
def print_messages(recipient, messages, sender, config):
    content = messages[-1]['content']
    if 'name' in messages[-1]:
        chat_interface.send(content, user=messages[-1]['name'], avatar=avatars[messages[-1]['name']], respond=False)
    else:
        chat_interface.send(content, user=recipient.name, avatar=avatars[recipient.name], respond=False)
    return False, None

# Register reply functions to capture and display messages with avatars
user_proxy.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
engineer.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
planner.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
executor.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
writer.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)

# Function to initiate the workflow
def submit_task(event):
    task = task_input.value
    if task:
        chat_interface.send(f"Task: {task}", user="System", respond=False)
        # Start the chat between Admin and Planner
        groupchat_result = user_proxy.initiate_chat(
            manager, message=f"Admin initiated the task: {task}"
        )
        print(groupchat_result)

submit_button.on_click(submit_task)

# Display Interface
tabs = pn.Tabs(
    ("Task Input", pn.Column(task_input, submit_button)),
    ("Agent Conversation", chat_interface),
    ("Results", pn.Column(sizing_mode="stretch_width")),
    margin=(20, 20),
)

# Display Layout
tabs.servable()
