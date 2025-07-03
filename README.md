# PDF Quiz Generator

A Django application that generates quizzes from PDF documents using LangChain and OpenAI.

## Features

- Upload a PDF document
- Choose the number of questions (1-50)
- Select difficulty level (easy, medium, hard)
- Generate a multiple-choice quiz based on the PDF content

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install django langchain langchain-community openai PyPDF2 python-dotenv
   ```

## Configuration

### OpenAI API Key

This application requires an OpenAI API key to function. You can set it up in one of the following ways:

1. **Environment Variable (Recommended)**:
   Set the `OPENAI_API_KEY` environment variable:

   ```bash
   # On Unix/Linux/macOS
   export OPENAI_API_KEY='your-api-key-here'

   # On Windows
   set OPENAI_API_KEY=your-api-key-here
   ```

   Alternatively, you can use the provided `.env` file:

   - Copy `.env.example` to `.env` (or use the existing `.env` file)
   - Edit the `.env` file and replace `your-api-key-here` with your actual OpenAI API key

2. **Settings File (For Development Only)**:
   You can directly set the API key in `project/settings.py`:

   ```python
   OPENAI_API_KEY = 'your-api-key-here'
   ```

   **Note**: This method is not recommended for production as it exposes your API key in the code.

## Running the Application

1. Start the Django development server:
   ```bash
   python manage.py runserver
   ```

2. Open your browser and navigate to `http://127.0.0.1:8000/`

3. Upload a PDF, select the number of questions and difficulty level, and generate your quiz!

## Notes

- The application limits the PDF content to approximately 3000 characters to avoid token overload with the OpenAI API.
- For larger PDFs, consider implementing a text splitter to process the document in chunks.
