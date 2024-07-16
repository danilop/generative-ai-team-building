#!/usr/bin/env python3
from typing import Any, Dict, List, Optional, Union

import argparse, base64, io, json, random, os, re, string

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from string import Template
from time import sleep

import boto3, botocore
import rich, rich.prompt
import pygame as pg

# PyGame custom events
NEW_TITLE_EVENT = pg.event.custom_type()
NEW_ICON_EVENT = pg.event.custom_type()

# Game visual parameters
FORCE_STRENGTH = 10
FRICTION = 0 # 0.005
MAX_SPEED = 10
ICON_SIZE = (96, 96)
EXPERT_SIZE = (96, 96)
TOP_LEFT_RIGHT_BORDER = 24
BOTTOM_BORDER = 48
TASK_WIDTH = 384

# Amazon Bedrock models and parameters
MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
# MODEL_ID = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
IMAGE_MODEL_ID = 'amazon.titan-image-generator-v1'

MAX_TOKENS = 4096
MAX_LOOPS = 10
TEMPERATURE = 1

MAX_JSON_RETRIES = 3
MAX_IMAGE_RETRIES = 10

TOOLS = [
    {
        "toolSpec": {
            "name": "python",
            "description": "Run a Python script. Use this for all math, date/time, or complex computations.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "The Python script to run.",
                        }
                    },
                    "required": ["script"],
                }
            },
        }
    },
]

# App configurations
APP_NAME = "TeamBuilding"
OUTPUT_PATH = "Output/"

# Prompts and Promt Templates Library

SYSTEM_PROMPT = (
    "Think step by step and add your notes between <thinking></thinking> tags."
)

LEAD_SYSTEM_PROMPT = "You lead a team of experts."

IMAGE_SYSTEM_PROMPT = "You are an experied pixelart artist working on 80s style retro videogames."

EXPERT_SYSTEM_PROMPT_TEMPLATE = Template("""You are $name.
$description.
Your role is $role.
Your expertise is $expertise.
"""
)

CREATE_GOAL_NAME_PROMPT_TEMPLATE = Template("""This is your goal:
<goal>
$goal
</goal>
Find a short (1-3 words), fun, and compelling name for a project to achieve this goal.
Avoid official product names, company names, trademarks, and acronyms in the goal name.
Output the results using this JSON format:
{
    "goal_name": "Goal Name",
    "icon_text_to_image_prompt": "A family-friendly text-to-image prompt to get an app icon for the goal, avoid writings and logos, no names, 8-bit retro 80s video game graphics",
}
Write the team JSON output between the <name></name> tags.
"""
)

CREATE_TEAM_PROMPT_TEMPLATE = Template("""This is your goal (named "$goal_name"):
<goal>
$goal
</goal>
Create a team of experts that can collaborate and find the best solution for your goal.
Use short job titles.
Output the experts using this JSON format:
[
    {
        "name": "Expert Name",
        "role": "Expert Job Title",
        "description": "Expert Description",
        "expertise": "Expertise Area",
        "text_to_image_prompt": "A visual family-friendly text-to-image prompt to show an image of the expert, avoid writings and logos, no names, 8-bit retro 80s video game graphics",
    },
    ...
]
Write the team JSON output between the <team></team> tags.
"""
)

REPLACE_IMAGE_PROMPT_TEMPLATE = Template("""Create a short, safer alternative to this text-to-image prompt preserving details, name, and description:
<old>
$prompt
</old>
Be sure the new text-to-image prompt contains a single visible human figure in the style of a 8-bit retro 80s video game sprite, family-friendly, pixel art.
Output the updated text-to-image prompt between the <new></new> tags.
"""
)

CREATE_PLAN_PROMPT_TEMPLATE = Template("""This is your goal (named "$goal_name"):
<goal>
$goal
</goal>
Using the following team of experts, create a detailed plan with multiple simpler tasks to achieve your goal in a step-by-step manner:
<team>
$team
</team>
Each task is assigned to exactly one expert.
Use short task names.
Output the plan using this JSON format:
[
    {
        "task": "Task Name",
        "description": "Detailed task description of what needs to be done",
        "assigned_expert": "Expert Name",
        "output": "Expected output of the task",
        "dependencies": ["Task Name", ...],
    },
    ...
]
Write the plan JSON output between the <plan></plan> tags.
"""
)

SOLVE_TASK_PROMPT_TEMPLATE = Template("""You are an expert. Your name is $assigned_expert
You are part of a team of experts:
<team>
$team
</team>
This is your goal (named "$goal_name"):
<goal>
$goal
</goal>
To achieve this goal, this is the overall plan:
<plan>
$plan
</plan>
These other tasks have already been solved:
<tasks>
$tasks
</tasks>
Now is your turn to solve this task:
<task>
$task
</task>
Based on your expertise, you can suggest to change something in an already solved task.
Write your solution using Markdown between the <output></output> tags.
Start your solution with the task name as title, for example:
<output>
# Task Name
...
</output>
If you need to create a file, use this syntax within the <output></output> tags:
<file fileName="full/path/filename.ext">
Content of the file...
</file>
Only create files where you can provide the full content.
"""
)

FEEDBACK_TASK_DESCRIPTION = """All tasks have been solved.
This is your opportunity to use your expertise to check if something is missing
or should be improved in the solution.
Focus on accuracy and clarity."""


bedrock_client_config = botocore.config.Config(
    read_timeout=900,
    connect_timeout=900,
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    }
)

bedrock_runtime = boto3.client(service_name="bedrock-runtime", config=bedrock_client_config)

goal = {} # Used to track the process towards the goal between the solve_goal() and play_game() functions

def init_goal():
    global goal
    for s in ['inputTokens', 'outputTokens', 'totalTokens', 'generatedImages']:
        goal[s] = 0

class ToolError(Exception):
    """Custom exception class for tool errors."""
    pass


def print_usage():
    global goal
    rich.print(f"Text model: input {goal['inputTokens']} / output {goal['outputTokens']} / total {goal['totalTokens']} tokens")
    rich.print(f"Image model: {goal['generatedImages']} images")

def call_bedrock(message_list: List[Dict[str, Any]], tool_list: List[Dict[str, Any]], system_prompt: str) -> Dict[str, Any]:
    """
    Call the Amazon Bedrock AI service with the given message and tool lists, and system prompt.
    
    Args:
        message_list (List[Dict[str, Any]]): List of messages to send to the AI service.
        tool_list (List[Dict[str, Any]]): List of tools available to the AI service.
        system_prompt (str): The system prompt to provide context for the AI service.
    
    Returns:
        Dict[str, Any]: The response from the AI service.
    """
    full_system_prompt = system_prompt + "\n" + SYSTEM_PROMPT
    while True:
        try:
            response = bedrock_runtime.converse(
                modelId=MODEL_ID,
                messages=message_list,
                system=[{"text": full_system_prompt}],
                inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": TEMPERATURE},
                toolConfig={"tools": tool_list},
            )
        except Exception as e:
            rich.print(e)
            rich.print(message_list)
            sleep(3)
        else:
            break

    if 'usage' in response:
        usage = response['usage']
        for s in ['inputTokens', 'outputTokens', 'totalTokens']:
            if s in usage:
                goal[s] += usage[s]

    print_usage()

    return response


def get_tool_result(tool_use_block: Dict[str, Any]) -> Optional[str]:
    """
    Execute the specified tool and return the result.

    Args:
        tool_use_block (Dict[str, Any]): The tool use block containing the tool name and input.

    Returns:
        Optional[str]: The result of the tool execution, or None if the tool is invalid.

    Raises:
        ToolError: If the specified tool is invalid.
    """
    tool_use_name = tool_use_block["name"]
    rich.print(f"Using tool {tool_use_name}")

    if tool_use_name == "python":
        script = tool_use_block["input"]["script"]
        rich.print(f"Script:\n{script}")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exec(script)
        output = stdout.getvalue()
        rich.print(f"Output: {output}")
        return output  # .decode().strip()
    else:
        raise ToolError(f"Invalid function name: {tool_use_name}")


def handle_response(response_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Handle the response message from the AI service.

    Args:
        response_message (Dict[str, Any]): The response message from the AI service.

    Returns:
        Optional[Dict[str, Any]]: A follow-up message to send back to the AI service, or None if no follow-up is needed.
    """
    response_content_blocks = response_message["content"]
    follow_up_content_blocks = []

    for content_block in response_content_blocks:
        if "toolUse" in content_block:
            tool_use_block = content_block["toolUse"]

            try:
                tool_result_value = get_tool_result(tool_use_block)
                if tool_result_value is not None:
                    follow_up_content_blocks.append(
                        {
                            "toolResult": {
                                "toolUseId": tool_use_block["toolUseId"],
                                "content": [{"json": {"result": tool_result_value}}],
                            }
                        }
                    )
            except ToolError as e:
                follow_up_content_blocks.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use_block["toolUseId"],
                            "content": [{"text": repr(e)}],
                            "status": "error",
                        }
                    }
                )

    if len(follow_up_content_blocks) > 0:
        follow_up_message = {
            "role": "user",
            "content": follow_up_content_blocks,
        }
        return follow_up_message
    else:
        return None


def run_loop(prompt: str, tool_list: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, Any]]:
    """
    Run a loop of interactions with the AI service.

    Args:
        prompt (str): The initial prompt to send to the AI service.
        tool_list (List[Dict[str, Any]]): List of tools available to the AI service.
        system_prompt (str): The system prompt to provide context for the AI service.

    Returns:
        List[Dict[str, Any]]: The list of messages exchanged with the AI service.
    """
    loop_count = 0
    continue_loop = True

    message_list = [{"role": "user", "content": [{"text": prompt}]}]

    while continue_loop:
        response = call_bedrock(message_list, tool_list, system_prompt)
        response_message = response["output"]["message"]
        message_list.append(response_message)

        loop_count = loop_count + 1
        if loop_count >= MAX_LOOPS:
            rich.print(f"Hit loop limit: {loop_count}")
            break

        follow_up_message = handle_response(response_message)
        if follow_up_message is None:
            continue_loop = False
        else:
            message_list.append(follow_up_message)

    return message_list


def get_text_between_tags(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """
    Extract the text between the specified start and end tags.

    Args:
        text (str): The input text containing the tags.
        start_tag (str): The opening tag.
        end_tag (str): The closing tag.

    Returns:
        Optional[str]: The text between the tags, or None if the tags are not found.
    """
    start_index = text.rfind(start_tag)
    if start_index == -1:
        return None
    start_index += len(start_tag)
    end_index = text.rfind(end_tag, start_index)
    if end_index == -1:
        return None
    return text[start_index:end_index]


def clean_file_name(text: str) -> str:
    """
    Clean the given text to create a valid file name.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text suitable for a file name.
    """
    # Replace non-alphanumeric characters with underscores, keep dots
    text = re.sub(r'[^a-zA-Z0-9._-]', '_', text)
    # Replace multiple consecutive underscores with a single underscore
    text = re.sub(r'_+', '_', text)
    # Remove leading/trailing underscores
    text = text.strip('_')

    return text


def write_file(goal_name: str, path: str, file_name: str, file_content: Union[str, bytes], binary: bool = False, overwrite: bool = False) -> str:
    """
    Write the given file content to a file.

    Args:
        goal_name (str): The name of the goal associated with the file.
        path (str): The path where the file should be written.
        file_name (str): The name of the file.
        file_content (Union[str, bytes]): The content to write to the file.
        binary (bool, optional): Whether the file content is binary. Defaults to False.
        overwrite (bool, optional): Whether to overwrite an existing file. Defaults to False.

    Returns:
        str: The full path of the written file.
    """
    full_path = os.path.join(OUTPUT_PATH, clean_file_name(goal_name), path)
    file_name = clean_file_name(file_name)

    os.makedirs(full_path, exist_ok=True)

    full_file_name = full_path + file_name
    if not overwrite:
        exists_index = 1
        while os.path.exists(full_file_name):
            new_file_name = f"{os.path.splitext(file_name)[0]}_{exists_index}{os.path.splitext(file_name)[1]}"
            new_full_file_name = full_path + new_file_name
            rich.print(f"File '{full_file_name}' already exists. Using '{new_full_file_name}'.")
            full_file_name = new_full_file_name
            exists_index += 1

    if binary:
        file_mode = "wb"
    else:
        file_mode = "w"
        
    rich.print(f"Writing file '{full_file_name}'")
    with open(full_file_name, file_mode) as f:
        f.write(file_content)

    return full_file_name


def extract_files(text: str, goal_name: str) -> None:
    """
    Extract file content from the given text and write them to files.

    Args:
        text (str): The input text containing file content.
        goal_name (str): The name of the goal associated with the files.
    """
    pattern = r'<file fileName="(.*?)">(.*?)</file>'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        file_path, file_content = match
        file_name = file_path.split("/")[-1]
        file_path = "Files/" + "/".join(file_path.split("/")[:-1]) + "/"
        file_path = file_path.replace("//", "/")
        write_file(goal_name, file_path, file_name, file_content, overwrite=True)


def process_prompt(prompt: str, system_prompt: str, output_tag_names: List[str], returnJson: bool = True) -> Dict[str, Any]:
    """
    Process the given prompt using the AI service and extract the results.

    Args:
        prompt (str): The prompt to send to the AI service.
        system_prompt (str): The system prompt to provide context for the AI service.
        output_tag_names (List[str]): The list of output tag names to extract from the AI response.
        returnJson (bool, optional): Whether to parse the output as JSON. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary containing the extracted results for each output tag.
    """

    retry = 0
    result_in_tag = None

    while retry < MAX_JSON_RETRIES and result_in_tag is None:

        messages = run_loop(prompt.strip(), TOOLS, system_prompt)

        # Concatenate in a string all assistant text messages
        result = ''
        for message in messages:
            if 'content' in message and 'role' in message:
                if message['role'] == 'assistant':
                    for content in message["content"]:
                        if 'text' in content:
                            result += content["text"]

        # Remove all <thinking>...</thinking> content from the result, using non-greedy matching
        result = re.sub(r'<thinking>.*?</thinking>', '', result, flags=re.DOTALL)

        results = {}
        for output_tag_name in output_tag_names:
            start_tag = f"<{output_tag_name}>"
            end_tag = f"</{output_tag_name}>"
            result_in_tag = get_text_between_tags(result, start_tag, end_tag)

            if returnJson:
                try:
                    result_in_tag = json.loads(result_in_tag) if result_in_tag else None
                except json.decoder.JSONDecodeError:
                    rich.print(f"Error: Invalid JSON\n{result_in_tag}")
                    retry += 1
                    rich.print(f"Retrying ({retry + 1}/{MAX_JSON_RETRIES})...")
                    result_in_tag = None
    
            results[output_tag_name] = result_in_tag

    return results


def generate_image(goal_name: str, file_path: str, file_name: str, prompt: str) -> str:
    """
    Generate an image using the Amazon Titan model based on the given prompt.

    Args:
        goal_name (str): The name of the goal associated with the image.
        file_path (str): The path where the generated image should be saved.
        file_name (str): The name of the image file.
        prompt (str): The text prompt to generate the image.

    Returns:
        str: The full path of the generated image file.
    """

    positive_prompt = prompt.strip() + ", in the style of a cute 8-bit retro 80s video game sprite, family-friendly, pixel art, without typography, no fonts, no text, no letters, no numbers, no logos"
    negative_prompt = "text, numbers, typography, logos, trademarks"

    # Format the request payload using the model's native structure.
    body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": positive_prompt,
            "negativeText": negative_prompt,
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "height": 512,
            "width": 512,
        },
    }

    retry = 0
    # Invoke the model with the request.
    while retry < MAX_IMAGE_RETRIES:
        try:
            response = bedrock_runtime.invoke_model(modelId=IMAGE_MODEL_ID, body=json.dumps(body))
            goal['generatedImages'] += 1
        except Exception as e:
            rich.print(e)
            prompt = body['textToImageParams']['text']
            results = process_prompt(
                REPLACE_IMAGE_PROMPT_TEMPLATE.substitute(prompt=prompt),
                IMAGE_SYSTEM_PROMPT,
                ["new"], returnJson=False
            )
            new_text_to_image_prompt = results['new']
            body['textToImageParams']['text'] = new_text_to_image_prompt
            rich.print("Updated text-to-image prompt:")
            rich.print(new_text_to_image_prompt)
            retry += 1
        else:
            break

    print_usage()

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract the image data.
    base64_image_data = model_response["images"][0]

    image_data = base64.b64decode(base64_image_data)

    full_file_name = write_file(goal_name, file_path, file_name, image_data, binary=True)

    return full_file_name


def pretty_print_json(json_object):
    rich.print_json(json.dumps(json_object))


def clean_markdown(markdown_text):
    return re.sub(r'\n\n\n+', '\n\n', markdown_text).strip(' \n\t')


def generate_toc(markdown_text):
    # Remove code blocks between triple backticks
    code_block_pattern = r'```.*?```'
    markdown_text_without_code_blocks = re.sub(code_block_pattern, '', markdown_text, flags=re.DOTALL)

    toc = []
    lines = markdown_text_without_code_blocks.split('\n')
    allowed_chars = string.ascii_letters + string.digits + '-'

    for line in lines:
        match = re.match(r'^(#{1,3})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            title = match.group(2)
            link = ''.join(char for char in title.lower().replace(' ', '-') if char in allowed_chars)

            if level <= 3:
                indent = '  ' * (level - 1)
                toc_entry = f"{indent}* [{title}](#{link})"
                toc.append(toc_entry)

    return '\n'.join(toc)


def get_opening():
    return f"# {goal['name']}\n\n{goal['description']}\n\n# Table of contents\n\n"


def add_toc(markdown_text):
    return get_opening() + generate_toc(markdown_text) + '\n\n' + markdown_text


def solve_goal():
    global goal

    rich.print("Finding the goal name...")
    goal['message'] = "Finding the goal name..."
    results = process_prompt(
        CREATE_GOAL_NAME_PROMPT_TEMPLATE.substitute(goal=goal['description']),
        LEAD_SYSTEM_PROMPT,
        ["name"]
    )
    goal['name'] = results['name']['goal_name']
    icon_text_to_image_prompt = results["name"]["icon_text_to_image_prompt"]
    pg.event.post(pg.event.Event(NEW_TITLE_EVENT))
    pretty_print_json(results['name'])

    rich.print(f"Generating icon for {goal['name']}...")
    goal['message'] = f"Generating icon for {goal['name']}..."
    file_name = f"{goal['name']}.png"
    goal['icon_file_name'] = generate_image(goal['name'], 'Icon/', file_name, icon_text_to_image_prompt)
    goal['icon_img'] = pg.image.load(goal['icon_file_name'])
    pg.event.post(pg.event.Event(NEW_ICON_EVENT))

    rich.print("Creating the team of experts...")
    goal['message'] = "Creating the team of experts..."
    results = process_prompt(
        CREATE_TEAM_PROMPT_TEMPLATE.substitute(goal_name=goal['name'], goal=goal),
        LEAD_SYSTEM_PROMPT,
        ["team"]
    )
    team = results['team']
    
    team_size = len(team)
    goal['team'] = []
    for index, expert in enumerate(team):
        rich.print(f"{index + 1}/{team_size} Generating image for {expert['name']} ({expert['role']})...")
        file_name = f"{expert['name']} {expert['role']}.png"
        text_to_image_prompt = f"{expert['text_to_image_prompt']}"
        pretty_print_json(expert)
        full_file_name = generate_image(goal['name'], 'Team/', file_name, text_to_image_prompt)
        expert['file_name'] = full_file_name
        goal['team'].append(expert)

    rich.print("Creating the plan...")
    for expert in goal['team']:
        expert['message'] = "Creating the plan"
    results = process_prompt(
        CREATE_PLAN_PROMPT_TEMPLATE.substitute(
            goal_name=goal['name'], goal=goal, team=team # Not using goal['team'] because it contains images
        ),
        LEAD_SYSTEM_PROMPT,
        ["team", "plan"],
    )
    goal['plan'] = results['plan']
    for expert in goal['team']:
       del expert['message']

    # Adding final feedback loop to the plan
    for expert in team:
        feedback_task = {
            "task": f"{expert['role']} Feedback",
            "description": FEEDBACK_TASK_DESCRIPTION,
            "assigned_expert": expert['name'],
            "output": f"Your feedback as a {expert['role']} to improve the current solution.",
            "dependencies": [], # No need because it is added at the end
        }
        goal['plan'].append(feedback_task)

    pretty_print_json(goal['plan'])

    # Finding the right task order considering dependencies
    goal['tasks'] = []
    completed_tasks = set()
    while len(completed_tasks) < len(goal['plan']):
        for task in goal['plan']:
            if task['task'] not in completed_tasks and set(task['dependencies']).issubset(completed_tasks):
                rich.print(f"Planning task {len(completed_tasks) + 1} / {len(goal['plan'])}")
                goal['tasks'].append(task)
                completed_tasks.add(task["task"])
                pretty_print_json(task)

    # Processing the tasks
    for index, task in enumerate(goal['tasks']):
        num_tasks = len(goal['tasks'])
        rich.print(f"Task {index + 1} / {num_tasks}")
        expert = next((e for e in goal['team'] if e['name'] == task['assigned_expert']), None)
        expert['message'] = f"{task['task']}"
        pretty_print_json(task)
        task['current'] = True
        if expert is None:
            raise ValueError(f"Expert '{task['assigned_expert']}' not found in team.")
        results = process_prompt(
            SOLVE_TASK_PROMPT_TEMPLATE.substitute(
                task=task,
                assigned_expert=task['assigned_expert'],
                team=team,
                goal_name=goal['name'],
                goal=goal,
                plan=goal['plan'],
                tasks=goal['tasks'],
            ),
            EXPERT_SYSTEM_PROMPT_TEMPLATE.substitute(
                name=expert['name'],
                description=expert['description'],
                role=expert['role'],
                expertise=expert['expertise'],
            ),
            ["output"],
            returnJson=False,
        )
        task['solution'] = results['output']
        del expert['message']
        del task['current']

        rich.print("Writing solution...")
        solution = ''
        for task in goal['tasks']:
            if 'solution' in task:
                solution += f"{task['solution']}\n\n"
        solution = clean_markdown(add_toc(solution))

        write_file(goal['name'], "", "solution.md", solution, overwrite=True)
        extract_files(solution, goal['name'])

    for expert in goal['team']:
        expert['message'] = "Bye!"

    sleep(3)

    for expert in goal['team']:
        del expert['message']

    goal['running'] = False
    rich.print("Bye!")


def apply_repulsive_force(p1, p2, width, height, strength = 1):
    force = strength * FORCE_STRENGTH
    dx = (p2['x'] - p1['x']) / width
    dy = (p2['y'] - p1['y']) / height
    if dx != 0:
        p1['vx'] -= force / dx
    if dy != 0:
        p1['vy'] -= force / dy


def apply_boundary_force(p, width, height, strength = 1):
    force = strength * FORCE_STRENGTH
    px = p['x'] / width
    py = p['y'] / height
    dx = px - TOP_LEFT_RIGHT_BORDER / width
    if dx != 0:
        p['vx'] += abs(force / dx)
    else:
        p['vx'] += force * width
    dx = 1 - px - (p['rect'].width + TOP_LEFT_RIGHT_BORDER) / width
    if dx != 0:
        p['vx'] -= abs(force / dx)
    else:
        p['vx'] -= force * width
    dy = py - TOP_LEFT_RIGHT_BORDER / height
    if dy != 0:
        p['vy'] += abs(force / dy)
    else:
        p['vy'] += force * height
    dy = 1 - py - (p['rect'].height + BOTTOM_BORDER) / height
    if dy != 0:
        p['vy'] -= abs(force / dy)
    else:
        p['vy'] -= force * height


def apply_movement_and_friction(p, width, height, dt):
    if abs(p['vx']) > MAX_SPEED:
        p['vx'] = MAX_SPEED * p['vx'] / abs(p['vx'])
    p['x'] += dt * p['vx']
    if p['x'] < 0:
        p['vx'] = -p['vx']
        p['x'] = 0
    if p['x'] + p['rect'].width >= width:
        p['vx'] = -p['vx']
        p['x'] = width - p['rect'].width
    p['vx'] *= (1 - FRICTION)
    if abs(p['vy']) > MAX_SPEED:
        p['vy'] = MAX_SPEED * p['vy'] / abs(p['vy'])
    p['y'] += dt * p['vy']
    if p['y'] < 0:
        p['vy'] = -p['vy']
        p['y'] = 0
    if p['y'] + p['rect'].height >= height:
        p['vy'] = -p['vy']
        p['y'] = height - p['rect'].height
    p['vy'] *= (1 - FRICTION)


def play_game():
    global goal

    pg.init()
    size = width, height = 1280, 768 # 16:9 ratio
    screen = pg.display.set_mode(size)
    pg.display.set_caption(APP_NAME)
    clock = pg.time.Clock()

    team_height = height
    team_width = width - TASK_WIDTH

    goal['running'] = True # Must be running before the beginning of the game event loop

    while goal['running']:

        dt = clock.tick(60) / 1000  # limits FPS to 60

        # Fill the screen with a color to wipe away anything from last frame
        my_color = (48, 25, 52)
        screen.fill(my_color)

        # Rendering
        if 'team' in goal and len(goal['team']) > 0:
            for expert in goal['team']:
                if 'img' not in expert:
                    # Load image and prepare initial position and speed
                    expert_img = pg.image.load(expert['file_name'])
                    expert_img = pg.transform.scale(expert_img, EXPERT_SIZE)
                    expert['img'] = expert_img
                    expert['rect'] = expert_img.get_rect()
                    expert['x'] = random.randint(0, team_width - expert['rect'].width)
                    expert['y'] = random.randint(0, team_height - expert['rect'].height)
                    expert['vx'] = 0
                    expert['vy'] = 0
                my_color = (255, 255, 128)
                my_font = pg.font.SysFont('Comic Sans MS', 14, True)
                name_surface = my_font.render(expert['name'], True, my_color)
                my_font = pg.font.SysFont('Comic Sans MS', 14)
                role_surface = my_font.render(expert['role'], True, my_color)
                screen.blit(expert['img'], (expert['x'], expert['y']))
                dy = expert['rect'].height
                dx = (expert['rect'].width - name_surface.get_width()) / 2
                screen.blit(name_surface, (expert['x'] + dx, expert['y'] + dy))
                dy += name_surface.get_height()
                dx = (expert['rect'].width - role_surface.get_width()) / 2
                screen.blit(role_surface, (expert['x'] + dx, expert['y'] + dy))
                if 'message' in expert:
                    message = f"{expert['message']}"
                    my_color = (128, 255, 128)
                    my_font = pg.font.SysFont('Comic Sans MS', 14, True)
                    message_surface = my_font.render(message, True, my_color)
                    dy += message_surface.get_height()
                    dx = (expert['rect'].width - message_surface.get_width()) / 2
                    screen.blit(message_surface, (expert['x'] + dx, expert['y'] + dy))
        else:
            if 'message' in goal:
                my_font = pg.font.SysFont('Comic Sans MS', 32)
                my_color = (255, 255, 128)
                message_surface = my_font.render(goal['message'], True, my_color)
                if 'icon_img' in goal:
                    icon_img_scaled = pg.transform.scale(goal['icon_img'], ICON_SIZE)
                    screen.blit(icon_img_scaled, ((width - icon_img_scaled.get_width()) / 2, (height - message_surface.get_height()) / 2 - BOTTOM_BORDER - icon_img_scaled.get_height()))
                screen.blit(message_surface,
                            ((width - message_surface.get_width()) / 2,
                            (height - message_surface.get_height()) / 2))
        if 'tasks' in goal:
            for index, task in enumerate(goal['tasks']):
                task_height = height / len(goal['plan'])
                if 'solution' in task:
                    my_color = (255, 255, 128)
                    background_color = (0, 0, 48)
                elif 'current' in task:
                    my_color = (128, 255, 128)
                    background_color = (48, 0, 0)
                else:
                    my_color = (128, 128, 255)
                    background_color = (0, 48, 0)
                my_font = pg.font.SysFont('Comic Sans MS', 14, True)
                rect_surface = pg.Surface((TASK_WIDTH, task_height))
                rect_surface.fill(background_color)
                task_surface = my_font.render(task['task'], True, my_color)
                dx = (TASK_WIDTH - task_surface.get_width()) / 2
                dy = (task_height - task_surface.get_height()) / 2
                rect_surface.blit(task_surface, (dx, dy))
                screen.blit(rect_surface, (width - TASK_WIDTH, height - task_height * (index + 1)))

        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                goal['running'] = False
            elif event.type == NEW_TITLE_EVENT:
                pg.display.set_caption(f"{APP_NAME} - {goal['name']}")
            elif event.type == NEW_ICON_EVENT:
                pg.display.set_icon(goal['icon_img'])

        # Movement
        if 'team' in goal:
            # Apply forces
            for expert in goal['team']:
                if 'img' not in expert:
                    continue
                for other_expert in goal['team']:
                    if 'img' in other_expert and expert != other_expert:
                        apply_repulsive_force(expert, other_expert, width, height)
                if 'message' in expert:
                    strength = 6 # To introduce movement when there is a message
                else:
                    strength = 4
                apply_boundary_force(expert, team_width, team_height, strength)
            # Movement + Friction
            for expert in goal['team']:
                if 'img' in expert:
                    apply_movement_and_friction(expert, team_width, team_height, dt);

        pg.display.update()

    pg.quit()


def get_goal():
    parser = argparse.ArgumentParser(description="Team Building – You're not alone to reach your goal")
    parser.add_argument('file', metavar='FILE', type=str, nargs='?',
                        help="Path to the goal description file")

    args = parser.parse_args()

    goal_description = None

    if args.file:
        file_path = args.file
        try:
            with open(file_path, 'r') as file:
                goal_description = file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except Exception as e:
            print(f"Error: {e}")
    else:
        while not goal_description:
            goal_description = rich.prompt.Prompt.ask("What is your goal? Enter quit or exit to leave").strip()
            if goal_description.lower() in ['quit', 'exit']:
                return None

    return goal_description


def main():
    global goal
    init_goal()

    rich.print("Welcome to Team Building! Your not alone to reach your goal.")
    goal['description'] = get_goal()
    if not goal['description']:
        return
    rich.print("This is my goal:")
    rich.print(f"[bold magenta]{goal['description']}[/bold magenta]")

    os.environ['SDL_VIDEO_CENTERED'] = '1' # Pygame is using SDL under the hood
    pg.init()
    size = width, heigth = 320, 240
    screen = pg.display.set_mode(size)
    pg.display.set_caption(APP_NAME)

    executor = ThreadPoolExecutor(max_workers=4)
    futures = []
    futures.append(executor.submit(solve_goal))
    play_game()
    for future in as_completed(futures):
        if future.exception():
            rich.print(future.result())

    executor.shutdown()


if __name__ == "__main__":
    main()
