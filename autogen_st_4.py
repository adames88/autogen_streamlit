import streamlit as st
import autogen
import asyncio
from tool.utils import get_openai_api_key, get_agentops_api_key
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

# Set up the OpenAI API key
get_openai_api_key()

# LLM Configuration
llm_config = {"model": "gpt-4o-mini", "temperature": 0, "seed": 1234}

# Define callback function for displaying messages in a dropdown (expander)
def display_callback(sender, recipient, message):
    sender_name = message.get('name', 'Unknown')
    content = message.get('content', '').strip().lower()

    # Skip system messages like task initiation
    if sender_name == "Admin" or "initiated the task" in content:
        return

    # Display only agent messages with name, avatar, and in a dropdown expander
    avatar = avatars.get(sender_name, '👤')
    with st.expander(f"{sender_name} (click to expand/collapse)", expanded=False):
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
        return processed_content

class TrackableUserProxyAgent(UserProxyAgent):
    def process_last_received_message(self, content):
        processed_content = super().process_last_received_message(content)
        return processed_content

class TrackableConversableAgent(ConversableAgent):
    def process_last_received_message(self, content):
        processed_content = super().process_last_received_message(content)
        return processed_content

# Define the agents with correct system prompts
user_proxy = TrackableUserProxyAgent(
    name="Admin",
    system_message="Admin. Give the task, and send instructions to writer to refine the financial report.",
    human_input_mode="NEVER",
    code_execution_config=False,
    is_termination_msg=lambda content: "TERMINATE" in content,
)

planner = TrackableConversableAgent(
    name="Planner",
    system_message="Planner. Given a task, please determine what information is needed to complete the task. "
                   "Please note that the information will all be retrieved using Python code. "
                   "Please only suggest information that can be retrieved using Python code. "
                   "After each step is done by others, check the progress and instruct the remaining steps. "
                   "If a step fails, try to workaround.",
    llm_config=llm_config,
    description="Planner. Given a task, determine what information is needed to complete the task. "
                "After each step is done by others, check the progress and instruct the remaining steps."
)

critic = TrackableConversableAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. "
                   "Check whether the plan includes adding verifiable info such as source URL.",
    llm_config=llm_config,
    description="Critic. Provide feedback for improvement for the planner and writer. "
                "Provide feedback for planner to improve overall plan. "
                "Provide feedback for writer to improve overall financial report."
)

engineer = TrackableAssistantAgent(
    name="Engineer",
    system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. 
Don't use a code block if it's not intended to be executed by the executor. Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor. Create graphs and plots. 
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try. 
Include code for saving plots, tables, graphs and any meaningful results. Always pass code you write to executor.""",
    llm_config=llm_config,
    code_execution_config=False,
    description="Engineer. An engineer that writes code based on the plan provided by the planner."
)

executor = TrackableConversableAgent(
    name="Executor",
    system_message="""Executor. You are a helpful AI assistant. Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute. 
1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself. 
2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly. Solve the task step by step if you need to. 
If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill. When using code, you must indicate the script type in the code block.""",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "coding",
        "use_docker": False,
    }
)

writer = TrackableConversableAgent(
    name="Writer",
    system_message="Writer. Please write a financial report in markdown format (with relevant titles) "
                   "and put the content in pseudo ```md``` code block. You take feedback from the admin "
                   "and refine your financial report.",
    llm_config=llm_config,
    description="Writer. Write financial report based on the code execution results and take feedback from the admin to refine the financial report."
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
