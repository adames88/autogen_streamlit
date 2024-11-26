import streamlit as st
import autogen
import asyncio
from tool.utils import get_openai_api_key
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

# Set up the OpenAI API key
get_openai_api_key()

# LLM Configuration
llm_config = {"model": "gpt-4o-mini", "temperature": 0, "seed": 1234}

# Callback function to display messages
def display_callback(sender, recipient, message):
    sender_name = message.get('name', 'Unknown')
    content = message.get('content', '').strip().lower()
    if sender_name == "Admin" or "initiated the task" in content:
        return
    avatar = avatars.get(sender_name, '👤')
    with st.expander(f"{sender_name} (click to expand/collapse)", expanded=False):
        st.markdown(f"**{sender_name}:** {content}")

# General function for handling replies
def print_messages(recipient, messages, sender, config):
    if "callback" in config and config["callback"] is not None:
        callback = config["callback"]
        callback(sender, recipient, messages[-1])
    print(f"Messages sent to: {recipient.name} | num messages: {len(messages)}")
    return False, None

# Avatars for agents
avatars = {
    "Planner": "🗓",
    "Engineer": "👩‍💻",
    "Executor": "🛠",
    "Writer": "✍",
    "Admin": "👨‍💼",
    "Critic": "🔍"
}

# Streamlit UI Setup
st.title("Agent Conversation and Task Management")
st.markdown("""
    <style>
    .stApp {
        background-color: #001d10;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Custom trackable classes
class TrackableAssistantAgent(AssistantAgent):
    def process_last_received_message(self, content):
        return super().process_last_received_message(content)

class TrackableUserProxyAgent(UserProxyAgent):
    def process_last_received_message(self, content):
        return super().process_last_received_message(content)

class TrackableConversableAgent(ConversableAgent):
    def process_last_received_message(self, content):
        return super().process_last_received_message(content)

# Define agents
user_proxy = TrackableUserProxyAgent(
    name="Admin",
    system_message="Admin. Give the task, and send instructions to writer to refine the financial report.",
    human_input_mode="NEVER",
    code_execution_config=False,
    is_termination_msg=lambda content: "TERMINATE" in content,
)

planner = TrackableConversableAgent(
    name="Planner",
    system_message="Planner. Given a task, determine required information for Python code retrieval and instruct next steps.",
    llm_config=llm_config,
)

critic = TrackableConversableAgent(
    name="Critic",
    system_message="Critic. Provide feedback for improvement for the planner and writer.",
    llm_config=llm_config,
)

engineer = TrackableAssistantAgent(
    name="Engineer",
    system_message="Engineer. Write Python code based on the plan provided by the planner.",
    llm_config=llm_config,
    code_execution_config=False,
)

executor = TrackableConversableAgent(
    name="Executor",
    system_message="Executor. Use coding and language skills to solve tasks.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "coding",
        "use_docker": False,
    },
)

writer = TrackableConversableAgent(
    name="Writer",
    system_message="Writer. Write a financial report in markdown format.",
    llm_config=llm_config,
)

# Define allowed speaker transitions
allowed_speaker_transitions_dict = {
    user_proxy: [planner, critic, engineer, executor, writer],
    planner: [user_proxy],
    critic: [user_proxy],
    engineer: [user_proxy],
    executor: [user_proxy],
    writer: [user_proxy],
}

# GroupChat setup
groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, critic, engineer, executor, writer],
    messages=[],
    max_round=50,
    allowed_or_disallowed_speaker_transitions=allowed_speaker_transitions_dict,
    speaker_transitions_type="allowed",
)

# GroupChatManager
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
    code_execution_config=False,
)

# Register reply functions
def register_agents_reply(agent):
    agent.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": display_callback})

for agent in [planner, critic, engineer, executor, writer, user_proxy]:
    register_agents_reply(agent)

# Initiate chat function
async def initiate_chat(task_input):
    await user_proxy.a_initiate_chat(manager, message=f"Admin initiated the task: {task_input}")

# Get user task input
task_input = st.chat_input("Enter your task (e.g., Retrieve stock prices for analysis)")

if task_input:
    st.write(f"**Task:** {task_input}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(initiate_chat(task_input))

# Placeholder for results
st.write("### Results")
