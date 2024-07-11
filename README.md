# Team Building

Welcome to Team Building!

Your not alone to reach your goal.

Pass a goal in input or via the command line.

Team Building will build:
- a name for the goal
- an icon for the goal
- a team of experts with their unique role, expertise, and icon
- a list of tasks (with dependencies with each other) to solve the goal
- a solution document with the output of the tasks
- the solution document includes a final feedback from each expert in the team

## Running a demo

First, you need to create a virtual environment to install dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A few sample goals are in the Demo directory. You can use them from the command line, for example:

```
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