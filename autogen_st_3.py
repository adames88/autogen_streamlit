import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import autogen
import asyncio
from tool.utils import get_openai_api_key, get_agentops_api_key
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

llm_config = {"model": "gpt-4o-mini","temperature": 0, "seed": 1234}

# Avatars for each agent (using emojis)
avatars = {
    "Planner": "🗓",
    "Engineer": "👩‍💻",
    "Executor": "🛠",
    "Writer": "✍"
}

# config_list_gpt4 = autogen.config_list_from_json(
#     llm_config,
#     filter_dict={
#         "model": ["gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
#     },
# )


# gpt4_config = {
#     "cache_seed": 42,  # change the cache_seed for different trials
#     "temperature": 0,
#     "config_list": config_list_gpt4,
#     "timeout": 120,
# }

def is_termination_msg(content) -> bool:
    have_content = content.get("content", None) is not None
    if have_content and "TERMINATE" in content["content"]:
        return True
    return False

user_proxy = TrackableUserProxyAgent(
    name="Admin",
    system_message="Admin."
    "Give the task, and send "
    "instructions to writer to refine the financial report.",
    human_input_mode="NEVER",
    code_execution_config=False,
    is_termination_msg=is_termination_msg,
)

planner = TrackableConversableAgent(
    name="Planner",
    system_message="Planner."
    "Given a task, please determine "
    "what information is needed to complete the task. "
    "Please note that the information will all be retrieved using"
    " Python code. Please only suggest information that can be "
    "retrieved using Python code. "
    "After each step is done by others, check the progress and "
    "instruct the remaining steps. If a step fails, try to "
    "workaround",
    llm_config=llm_config,
    description="Planner. Given a task, determine what "
    "information is needed to complete the task. "
    "After each step is done by others, check the progress and "
    "instruct the remaining steps"
    ""
)

critic = TrackableConversableAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
    llm_config=llm_config,
    description="Critic."
    "A Critic that prvides feedback for improvement for the planner and writer."
    "Provide feedback for planner to improve overall plan."
    "Provide feedback for writer to improve overall financial report."
)

engineer = TrackableAssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    code_execution_config=False,
    system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor. Create graphs and plots.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
Include code for saving plots, tables, graphs and any meaningful results.
Always pass code you write to executor.
""",
    description="Engineer."
    "An engineer that writes code based on the plan "
    "provided by the planner.",
)

executor = TrackableConversableAgent(
    name="Executor",
    system_message="""Executor. You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done.""",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "coding",
        "use_docker": False,
    },
)

writer = TrackableConversableAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="Writer." 
    "Please write a finanial report in markdown format (with relevant titles)"
    " and put the content in pseudo ```md``` code block. "
    "You take feedback from the admin and refine your financial report.",
    description="Writer."
    "Write financial report based on the code execution results and take "
    "feedback from the admin to refine the financial report."
)

allowed_speaker_transitions_dict = {
    user_proxy: [planner, critic, engineer, executor,writer],
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
