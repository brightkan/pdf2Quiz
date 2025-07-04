# Project Guidelines: PDF Quiz Generator (Django + LangChain + OpenAI)

## Overview
This app allows users to upload a PDF, select the number of questions and difficulty level, and receive a generated quiz based on the content using LangChain and OpenAI. The primary goal is to help students ace their real exams by providing targeted practice with customized questions that match their study materials.

## Key Goals
celery-1  |     run_checks()
celery-1  |   File "/usr/local/lib/python3.11/site-packages/django/core/checks/registry.py", line 89, in run_checks
celery-1  |     new_errors = check(app_configs=app_configs, databases=databases)
celery-1  |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
celery-1  |   File "/usr/local/lib/python3.11/site-packages/django/core/checks/urls.py", line 16, in check_url_config
celery-1  |     return check_resolver(resolver)
celery-1  |            ^^^^^^^^^^^^^^^^^^^^^^^^
celery-1  |   File "/usr/local/lib/python3.11/site-packages/django/core/checks/urls.py", line 26, in check_resolver
celery-1  |     return check_method()
celery-1  |            ^^^^^^^^^^^^^^
celery-1  |   File "/usr/local/lib/python3.11/site-packages/django/urls/resolvers.py", line 531, in check
celery-1  |     for pattern in self.url_patterns:
celery-1  |                    ^^^^^^^^^^^^^^^^^
celery-1  |   File "/usr/local/lib/python3.11/site-packages/django/utils/functional.py", line 47, in __get__
celery-1  |     res = instance.__dict__[self.name] = self.func(instance)
celery-1  |                                          ^^^^^^^^^^^^^^^^^^^
celery-1  |   File "/usr/local/lib/python3.11/site-packages/django/urls/resolvers.py", line 718, in url_patterns
celery-1  |     patterns = getattr(self.urlconf_module, "urlpatterns", self.urlconf_module)
celery-1  |                        ^^^^^^^^^^^^^^^^^^^
celery-1  |   File "/usr/local/lib/python3.11/site-packages/django/utils/functional.py", line 47, in __get__
celery-1  |     res = instance.__dict__[self.name] = self.func(instance)
celery-1  |                                          ^^^^^^^^^^^^^^^^^^^
celery-1  |   File "/usr/local/lib/python3.11/site-packages/django/urls/resolvers.py", line 711, in urlconf_module
celery-1  |     return import_module(self.urlconf_name)
celery-1  |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
celery-1  |   File "/usr/local/lib/python3.11/importlib/__init__.py", line 126, in import_module
celery-1  |     return _bootstrap._gcd_import(name[level:], package, level)
celery-1  |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
celery-1  |   File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
celery-1  |   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
celery-1  |   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
celery-1  |   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
celery-1  |   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
celery-1  |   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
celery-1  |   File "/app/project/urls.py", line 5, in <module>
celery-1  |     from core.views import quiz_view, interactive_quiz, submit_quiz, quiz_progress, check_task_status
celery-1  |   File "/app/core/views.py", line 13, in <module>
celery-1  |     from .quiz_generator import generate_quiz_from_pdf
celery-1  |   File "/app/core/quiz_generator.py", line 295
celery-1  |     topic_pattern = re.compile(r'^topic\s*:\s*(.+)
celery-1  |                                ^
celery-1  | SyntaxError: unterminated string literal (detected at line 295)
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
