import streamlit as st
import autogen
import asyncio
from tool.utils import get_openai_api_key, get_agentops_api_key
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

# Set up the OpenAI API key
get_openai_api_key()

# LLM Configuration
llm_config = {"model": "gpt-4o-mini", "temperature": 0, "seed": 1234}

# Define callback function for displaying messages in Streamlit
def display_callback(sender, recipient, message):
    sender_name = message.get('name', 'Unknown')
    content = message.get('content', '').strip().lower()

    # Skip system messages like task initiation
    if sender_name == "Admin" or "initiated the task" in content:
        return

    # Display only agent messages with name and avatar
    avatar = avatars.get(sender_name, '👤')
    with st.chat_message(sender_name, avatar=avatar):
        # Add the agent's name next to the avatar
        st.markdown(f"**{sender_name}:** {content}")

# General function to handle agent replies and invoke the callback
def print_messages(recipient, messages, sender, config):
    if "callback" in config and config["callback"] is not None:
        callback = config["callback"]
        callback(sender, recipient, messages[-1])  # Display the latest message

    print(f"Messages sent to: {recipient.name} | num messages: {len(messages)}")
    return False, None  # Ensure agent communication flow continues

# Avatars for each agent
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

# Custom trackable classes that hook into process_last_received_message
class TrackableAssistantAgent(AssistantAgent):
    def process_last_received_message(self, content):
        processed_content = super().process_last_received_message(content)
        # Handle any specific processing for AssistantAgent here if needed
        return processed_content

class TrackableUserProxyAgent(UserProxyAgent):
    def process_last_received_message(self, content):
        processed_content = super().process_last_received_message(content)
        # Handle any specific processing for UserProxyAgent here if needed
        return processed_content

class TrackableConversableAgent(ConversableAgent):
    def process_last_received_message(self, content):
        processed_content = super().process_last_received_message(content)
        # Handle any specific processing for ConversableAgent here if needed
        return processed_content

# Define the agents
user_proxy = TrackableUserProxyAgent(
    name="Admin",
    system_message="Admin. Provide task details to planner and feedback to writer.",
    human_input_mode="NEVER",
    code_execution_config=False,
)

planner = TrackableConversableAgent(
    name="Planner",
    system_message="Planner. Outline the task plan using Python-based retrieval methods.",
    llm_config=llm_config,
    description="Planner agent tasked with outlining a Python-executable plan."
)

critic = TrackableConversableAgent(
    name="Critic",
    system_message="Critic. Review the task plan and provide feedback.",
    llm_config=llm_config,
    description="Critic agent responsible for reviewing and improving plans."
)

engineer = TrackableAssistantAgent(
    name="Engineer",
    system_message="Engineer. Write and execute Python code for the task.",
    llm_config=llm_config,
    description="Engineer agent responsible for writing and executing code."
)

executor = TrackableConversableAgent(
    name="Executor",
    system_message="Executor. Solve tasks using Python code and review results.",
    human_input_mode="NEVER",
    llm_config=llm_config,
)

writer = TrackableConversableAgent(
    name="Writer",
    system_message="Writer. Compile results into a markdown report.",
    llm_config=llm_config,
)

# Define allowed agent transitions
allowed_speaker_transitions_dict = {
    user_proxy: [planner, critic, engineer, executor, writer],
    planner: [user_proxy],
    critic: [user_proxy],
    engineer: [user_proxy],
    executor: [user_proxy],
    writer: [user_proxy],
}

# Group Chat Setup
groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, critic, engineer, executor, writer],
    messages=[],
    max_round=50,
    allowed_or_disallowed_speaker_transitions=allowed_speaker_transitions_dict,
    speaker_transitions_type="allowed",
)

# Group Chat Manager
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
    code_execution_config=False,
)

# Register reply functions with callback for displaying messages
def register_agents_reply(agent):
    agent.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": display_callback})

for agent in [planner, critic, engineer, executor, writer, user_proxy]:
    register_agents_reply(agent)

# Function to initiate the chat
async def initiate_chat(task_input):
    await user_proxy.a_initiate_chat(manager, message=f"Admin initiated the task: {task_input}")

# Get user task input
task_input = st.chat_input("Enter your task (e.g., Retrieve stock prices for analysis)")

if task_input:
    st.write(f"**Task:** {task_input}")
    
    # Create an event loop and run the chat initiation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(initiate_chat(task_input))

# Placeholder for results
st.write("### Results")
