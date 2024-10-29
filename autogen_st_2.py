import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import autogen
import asyncio
from tool.utils import get_openai_api_key
from autogen.coding import LocalCommandLineCodeExecutor
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

# Custom trackable classes to display messages in Streamlit chat
class TrackableAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


class TrackableUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


class TrackableConversableAgent(ConversableAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)

# Set up the OpenAI API key
get_openai_api_key()

llm_config = {"model": "gpt-4-turbo"}

# Avatars for each agent (using emojis)
avatars = {
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
    timeout=120,
    work_dir="coding",
)

executor = TrackableUserProxyAgent(
    name="Executor",
    description="Execute the code written by the Engineer and report the result. Execute multiple steps if provided."
                "When you have fully completed the execution of code successfully, give the results and information to the writer."
                "Save graph, visualisation plots and data in the current directory and share it with the writer.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 3, "executor": executor_func},
)

# Planner with enhanced multi-step handling
planner = TrackableConversableAgent(
    name="Planner",
    system_message=(
        "You are responsible for planning the task. Break it down into steps and coordinate with Engineer for code."
        "If steps fail, guide the agents to retry. Never ask the engineer to run code, only provide plan for the engineer to write the code."
    ),
    description="Plan and delegate tasks in a step-by-step manner and ensure successful task completion.",
    llm_config=llm_config,
)

# Engineer to write code based on Planner instructions
engineer = TrackableAssistantAgent(
    name="Engineer",
    system_message=(
        "Write python code based on the Planner's instructions. Communicate with the Executor to run the code. "
        "Never run code, only write python code and pass it to the executor to run."
    ),
    description="Write and iterate code for stock price retrieval and analysis based on Planner's instructions.",
    llm_config=llm_config,
)

# Writer Agent
writer = TrackableConversableAgent(
    name="Writer",
    system_message="Write the final report after analysis. Refine based on feedback."
                   "When you have written your final report save it in the current directory as a markdown file."
                   "Present the information in a clean, professional, user-friendly, and aesthetically pleasing presentation.",
    description="Write and refine reports based on the results of the analysis.",
    llm_config=llm_config,
)

# Define Admin Agent (user_proxy)
user_proxy = TrackableUserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

# # managing workflow for agents
# def state_transition(last_speaker, groupchat):
#     messages = groupchat.messages

#     if last_speaker is user_proxy:
#         # init -> retrieve
#         return planner
#     elif last_speaker is planner:
#         # retrieve: action 1 -> action 2
#         return engineer
#     elif last_speaker is engineer:
#         # retrieve: action 1 -> action 2
#         return executor
#     elif last_speaker is executor:
#         if messages[-1]["content"] == "exitcode: 1":
#             # retrieve --(execution failed)--> retrieve
#             return engineer
#         elif 'possibly delisted' or 'no price data found' in messages[-1]["content"]:
#               return engineer

#         else:
#             # retrieve --(execution success)--> writer
#             return writer
#     elif last_speaker == "writer":
#         # research -> end
#         return None

# Group Chat for Agents
groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, writer, executor, planner],
    messages=[],
    max_round=50
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Streamlit UI Setup
st.title("Agent Conversation and Task Management")

# Set background color to #001d10 using native markdown
st.markdown(
    """
    <style>
    .stApp {
        background-color: #001d10;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Track whether Admin is waiting for input
if "admin_waiting" not in st.session_state:
    st.session_state["admin_waiting"] = False
    st.session_state["admin_prompt"] = ""  # To store admin's prompt

# Function to display messages in Streamlit with avatars
def print_messages(recipient, messages, sender, config):
    content = messages[-1]['content']
    user_name = messages[-1].get('name', sender.name)
    user_avatar = avatars.get(user_name, "")
    
    # Alternating messages between left and right based on the agent
    if user_name in ["Admin", "Planner"]:
        st.chat_message("assistant").write(f"{user_avatar} **{user_name}:** {content}")
    else:
        st.chat_message("user").write(f"{user_avatar} **{user_name}:** {content}")

    # Handle Admin waiting for user input
    if user_name == "Admin" and "Provide feedback" in content:
        st.session_state["admin_waiting"] = True
        st.session_state["admin_prompt"] = content  # Store Admin's prompt for display in the UI

    return False, None

# Register reply functions to capture and display messages with avatars
engineer.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
planner.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
executor.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
writer.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)

# Function to initiate the workflow asynchronously
async def initiate_chat(task_input):
    await user_proxy.a_initiate_chat(manager, message=f"Admin initiated the task: {task_input}")

# Get user task input
task_input = st.chat_input("Enter your task (e.g., Retrieve stock prices for analysis)", key="task_input_key")  # Unique key provided
if task_input:
    st.write(f"**Task:** {task_input}")

    # Create an event loop and run the chat initiation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(initiate_chat(task_input))

# Admin feedback input when waiting for user input
if st.session_state["admin_waiting"]:
    st.write(f"**Admin is requesting feedback:** {st.session_state['admin_prompt']}")  # Display Admin's prompt
    admin_feedback = st.text_input("Admin is asking for feedback. Please provide your input:", key="admin_feedback_key")  # Unique key provided

    if admin_feedback:
        # Send Admin feedback to the backend
        groupchat_result = user_proxy.initiate_chat(
            manager, message=f"{admin_feedback}"
        )
        st.write(f"**Admin Response Sent:** {admin_feedback}")
        st.session_state["admin_waiting"] = False

# Placeholder for results
st.write("### Results")
