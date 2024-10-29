import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import autogen
import asyncio
import datetime
from tool.utils import get_openai_api_key, get_agentops_api_key
from autogen.coding import LocalCommandLineCodeExecutor
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

# Custom trackable classes to display messages in Streamlit chat
class TrackableAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with st.chat_message(sender.name):
            st.markdown(f"**{timestamp}:** {message}")
        return super()._process_received_message(message, sender, silent)


class TrackableUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with st.chat_message(sender.name):
            st.markdown(f"**{timestamp}:** {message}")
        return super()._process_received_message(message, sender, silent)


class TrackableConversableAgent(ConversableAgent):
    def _process_received_message(self, message, sender, silent):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with st.chat_message(sender.name):
            st.markdown(f"**{timestamp}:** {message}")
        return super()._process_received_message(message, sender, silent)

# Set up the OpenAI API key
get_openai_api_key()

llm_config = {"model": "gpt-4o-mini", "temperature": 0, "seed": 1234}

def is_termination_msg(content) -> bool:
    have_content = content.get("content", None) is not None
    if have_content and "TERMINATE" in content["content"]:
        return True
    return False

user_proxy = TrackableUserProxyAgent(
    name="Admin",
    system_message="Admin. Give the task, and send instructions to writer to refine the financial report.",
    human_input_mode="NEVER",
    code_execution_config=False,
    is_termination_msg=is_termination_msg,
)

planner = TrackableConversableAgent(
    name="Planner",
    system_message="Planner. Given a task, determine what information is needed to complete the task. Retrieve all info using Python code. After each step is done, check progress and instruct further.",
    llm_config=llm_config,
    description="Planner. Given a task, determine what info is needed and guide the process."
)

critic = TrackableConversableAgent(
    name="Critic",
    system_message="Critic. Double-check plans, code, and provide feedback for improvement. Ensure tasks include verifiable info such as source URLs.",
    llm_config=llm_config,
    description="Critic. Provides feedback for the Planner and Writer to improve their work."
)

engineer = TrackableAssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    code_execution_config=False,
    system_message="""Engineer. Follow approved plans. Write complete Python/shell code for tasks. Ensure the code is error-free and provide visualizations (graphs, tables).""",
    description="Engineer. Writes and executes code as per Planner's guidance."
)

executor = TrackableConversableAgent(
    name="Executor",
    system_message="""Executor. Execute tasks and write Python/shell code when required. Provide code results or analysis, checking if the solution is correct.""",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 3, "work_dir": "coding", "use_docker": False},
)

writer = TrackableConversableAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="Writer. Write a financial report in markdown format. Take feedback from Admin to refine the report.",
    description="Writer. Writes financial reports based on code execution results."
)

allowed_speaker_transitions_dict = {
    user_proxy: [planner, critic, engineer, executor, writer],
    planner: [user_proxy],
    critic: [user_proxy],
    engineer: [user_proxy],
    executor: [user_proxy],
    writer: [user_proxy],
}

groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, writer, planner, executor, critic], messages=[], max_round=50,
    allowed_or_disallowed_speaker_transitions=allowed_speaker_transitions_dict,
    speaker_transitions_type="allowed",
)

# Create the manager
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
    code_execution_config=False,
    is_termination_msg=is_termination_msg,
)

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

# Avatars for each agent (using emojis)
avatars = {
    "Planner": "🗓",
    "Engineer": "👩‍💻",
    "Executor": "🛠",
    "Writer": "✍",
    "Admin": "👨‍💼",
    "Critic": "🔍"
}

# Function to display messages in Streamlit with avatars and timestamps
def print_messages(recipient, messages, sender, config):
    content = messages[-1]['content']
    user_name = messages[-1].get('name', sender.name)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Displaying agents' message with corresponding avatar and timestamp
    if user_name:
        st.chat_message(user_name, avatar=avatars[user_name]).write(f"{timestamp}: {content}")

    return False, None

# Register reply functions to capture and display messages with avatars
engineer.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
planner.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
executor.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
critic.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
writer.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)
user_proxy.register_reply([autogen.Agent, None], reply_func=print_messages, config=None)

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

# Adding functionality to resume previous chat
if st.button("Resume Last Conversation"):
    st.write("Resuming previous group chat session...")
    groupchat.resume_last_chat()

# Placeholder for results
st.write("### Results")
