# Project Guidelines: PDF Quiz Generator (Django + LangChain + OpenAI)

## Overview
This app allows users to upload a PDF, select the number of questions and difficulty level, and receive a generated quiz based on the content using LangChain and OpenAI.

## Key Goals
- Upload a PDF document.
- Choose number of questions (1–50).
- Select difficulty level: easy, medium, hard.
- Generate a multiple-choice quiz using LangChain and OpenAI.
- Display quiz in a clean, readable format.

## Tech Stack
- Backend: Django
- AI: LangChain + OpenAI API
- PDF Parsing: `PyPDFium2Loader` (preferred for speed) or `PyPDF2`

## Design Rules
- Keep the frontend simple (Django forms/templates)
- Limit input text to ~3000 characters to avoid token overload
- Use `langchain_community` imports (LangChain v0.1+)

## Prompts
Prompt for quiz generation:


## LangChain Notes
- Use `ChatOpenAI` from `langchain.chat_models`
- If needed, chain `PyPDFLoader` with a splitter for longer PDFs

## Structure Suggestions
- `forms.py`: form for PDF, num_questions, difficulty
- `views.py`: handles form, calls `quiz_generator.py`
- `quiz_generator.py`: handles PDF parsing and quiz generation
- `templates/`: simple `form.html` and `quiz.html`

## Installation

### Standard Installation
```bash
pip install django langchain langchain-community openai PyPDF2
```

### Docker Development Environment
We use Docker for development to ensure a consistent environment across all developers. The Docker setup includes:

- **web**: Django application with hot-reloading
- **db**: PostgreSQL database
- **redis**: Redis for caching and Celery

To get started with Docker:

1. Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
2. Create a `.env` file with your OpenAI API key
3. Build and start the containers:
   ```bash
   docker-compose up --build
   ```
4. Access the application at http://localhost:8000
5. Run Django commands inside the container:
   ```bash
   docker-compose exec web python manage.py migrate
   ```

For more details, refer to DOCKER.md in the project root.
