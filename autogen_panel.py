import autogen
import panel as pn
import asyncio
from utils import get_openai_api_key

get_openai_api_key()

llm_config = {"model": "gpt-4-turbo"}

# Define Agents
user_proxy = autogen.ConversableAgent(
    name="Admin",
    system_message="Give the task, and send instructions to writer to refine the financial report.",
    code_execution_config=False,
    llm_config=llm_config,
    human_input_mode="Never",
)

planner = autogen.ConversableAgent(
    name="Planner",
    system_message="Given a task, determine what information is needed to complete the task. "
                   "All information should be retrievable via Python code. "
                   "After each step, check progress and instruct the next steps. Handle failures gracefully.",
    description="Determine information needed for task completion and manage progress after each step.",
    llm_config=llm_config,
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    description="Write code based on the plan provided by the planner.",
)

writer = autogen.ConversableAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="Writer. Write financial report in markdown format (with relevant titles) "
                   "and put the content in a ```md``` code block. Refine based on Admin feedback.",
    description="Write a financial report and refine it based on Admin's feedback.",
)

executor = autogen.ConversableAgent(
    name="Executor",
    description="Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "coding",
        "use_docker": False,
    },
)

# Define GroupChat
groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, writer, executor, planner],
    messages=[],
    max_round=50,
    allowed_or_disallowed_speaker_transitions={
        user_proxy: [engineer, writer, executor, planner],
        engineer: [user_proxy, executor],
        writer: [user_proxy, planner],
        executor: [user_proxy, engineer, planner],
        planner: [user_proxy, engineer, writer],
    },
    speaker_transitions_type="allowed",
)

manager = autogen.GroupChatManager(
    groupchat=groupchat, llm_config=llm_config
)

avatar = {user_proxy.name: "👨‍💼", engineer.name: "👩‍💻", writer.name: "✍", planner.name: "🗓", executor.name: "🛠"}

# Function to display messages in Panel UI
def print_messages(recipient, messages, sender, config):
    content = messages[-1]['content']
    if 'name' in messages[-1]:
        chat_interface.send(content, user=messages[-1]['name'], avatar=avatar[messages[-1]['name']], respond=False)
    else:
        chat_interface.send(content, user=recipient.name, avatar=avatar[recipient.name], respond=False)
    return False, None

# Register replies for all agents
user_proxy.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": None})
engineer.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": None})
writer.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": None})
planner.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": None})
executor.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": None})

# Panel UI setup
pn.extension(design="material")

initiate_chat_task_created = False

async def delayed_initiate_chat(agent, recipient, message):
    global initiate_chat_task_created
    initiate_chat_task_created = True
    await asyncio.sleep(2)
    await agent.a_initiate_chat(recipient, message=message)

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    global initiate_chat_task_created
    global input_future

    if not initiate_chat_task_created:
        asyncio.create_task(delayed_initiate_chat(user_proxy, manager, contents))
    else:
        if input_future and not input_future.done():
            input_future.set_result(contents)

chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send("Send a message!", user="System", respond=False)

# Panel input for task and submit button
task_input = pn.widgets.TextInput(name="Enter your task", placeholder="E.g., Write a financial report about Nvidia's stock price performance.")
submit_button = pn.widgets.Button(name="Submit Task", button_type="primary")

# Function to start the conversation based on user input
def submit_task(event):
    task = task_input.value
    if task:
        chat_interface.send(f"Task: {task}", user="System", respond=False)
        asyncio.create_task(delayed_initiate_chat(user_proxy, manager, task))

submit_button.on_click(submit_task)

# Layout for the app
app_layout = pn.Column(
    task_input,
    submit_button,
    chat_interface,
)

app_layout.servable()
