# Team Building

Welcome to Team Building! Your not alone to reach your goal.

Pass a goal in input or via the command line.

Team Building will:

- Find a name for the goal
- Design an icon for the goal
- Assemble a team of experts with unique roles, expertise, and images
- Prepare a list of tasks (with dependencies with each other) to solve the goal
- Build a solution document with the output of the tasks

The solution document will include a final round of feedback from each expert in the team.

## Running a quick demo

First, you should create a virtual environment to install dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A few sample goals are in the `Demo` directory. You can use them from the command line, for example:

```sh
./team-building.py ./Demo/books-app.txt
```

The content of the books-app.txt file is a textual descriiption of what you want to achieve:

```
Write a web app to store all the info about my books, including whether they have been lent to someone.
Start by collecting requirements and user stories.
Use Python Flask in the backend, React and Typescript in the frontend. Use PostgreSQL for the database.
Use AWS for deployment, with Amazon ECS for the backend and Amazon Aurora for the database.
Manage the infrastructure as code with AWS CDK.
Provide a complete solution including code and deployment instructions.
Make sure to include input validation, error handling, and logging.
Use an organized structure for files so that each group is in a different folder. Use different file names when possible.
```

## How it works

To display graphics and animations, Team Building uses [Pygame Community Edition](https://pyga.me/).

To invoke a generative AI model, the [Amazon Bedrock](https://aws.amazon.com/bedrock/) [Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html) are invoked directly using the [AWS SDK for Pyhon (Boto3)](https://aws.amazon.com/sdk-for-python/).

The text model to use is specified in `MODEL_ID`. Uncomment the line using the model of your choice. By default, it uses [Anthropic Claude 3 Sonnet](https://aws.amazon.com/bedrock/claude/).

The text-to-image model is specified in `IMAGE_MODEL_ID`. It currently uses [Amazon Titan Image Generator](https://aws.amazon.com/bedrock/titan/).

When started, Team Building reads the description of a goal you want to accomplish. The goal can be in a text file passed via the command line or typed interactived in the terminal.

Then, Team Building will proceed automously with the following steps:

1. Find a nice and short ame and a visual description for the goal.

2. Use the visual descirption to generate an icon for the goal. The icon is displaied in the window and assumed by the Team Building application.

3. The core idea is to create a team of experts that can help solve the goal The experts have differect specialties to be able to cover all the tasks required to accomplish the goal.

The output the experts has this JSON format:

```json
[
    {
        "name": "Expert Name",
        "role": "Expert Job Title",
        "description": "Expert Description",
        "expertise": "Expertise Area",
        "text_to_image_prompt": "...",
    },
    ...
]
```

4. Each expert description includes a text-to-image prompt that is used to generate a small image of the expert that is used in the visualization.

5. To solve the goal, the model is asked to generate a detailed plan with a list of tasks. Each tasks is assigned to one of the experts and includes dependencies of which other tasks in the plan need to be solved before it can start.

The output the plan has this JSON format:

```json
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
```

6. At this point Team Building will find the right order to run the tasks and will use the identity of each expert to solve the assigned tasks.

7. The output of each task is added to a Markdown solution document that is passed in input for each task.

For example, this is part of the prompt used to run a task:

```xml
You are part of a team of experts:
<team>...</team>
This is your goal (named "$goal_name"):
<goal>...</goal>
To achieve this goal, this is the overall plan:
<plan>...</plan>
These other tasks have already been solved:
<tasks>...</tasks>
Now is your turn to solve this task:
<task>...</task>
```

8. All outputs, including the goal icon and the expert images, are written in an `Output` directory in a folder names as the goal:

```xml
Output/<goalName>
```

9. If a tasks output includes a file, it is written in the `Output/<goal name>/Files` path.

To create a file, the model is asked to use this syntax within their output:

```xml
<file fileName="full/path/filename.ext">
Content of the file...
</file>
```

10. At the end of the tasks, a final round of feedback from each expert completes the output.

11. The final output is written in Markkdown format in the `Output/solution.md` file. The file can be used with any tool (for example, [Pandoc](https://pandoc.org/)) to convert Markdown to other formats such as HTML or Word files.
