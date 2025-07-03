import io
import re
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
import openai
from django.conf import settings
from .models import TokenUsage, UploadedPDF, Quiz, Question

def extract_text_from_pdf(pdf_file, pdf_obj=None):
    """
    Extract text from a PDF file and cache it if a pdf_obj is provided.
    If the text is already cached, return it instead of re-extracting.
    """
    # If pdf_obj is provided and has cached text, return it
    if pdf_obj and pdf_obj.extracted_text:
        return pdf_obj.extracted_text

    # Otherwise extract the text
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()

    # Cache the extracted text if pdf_obj is provided
    if pdf_obj:
        pdf_obj.save_extracted_text(text)

    return text

def chunk_text(text, max_chunk_size=1000, overlap=100):
    """
    Split text into chunks of maximum size with overlap between chunks.
    This helps maintain context between chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chunk_size, text_length)

        # If we're not at the end of the text, try to find a good breaking point
        if end < text_length:
            # Try to find the last period, question mark, or exclamation point followed by a space
            last_period = max(
                text.rfind('. ', start, end),
                text.rfind('? ', start, end),
                text.rfind('! ', start, end)
            )

            if last_period != -1:
                end = last_period + 2  # Include the period and space

        chunks.append(text[start:end])

        # Move the start position for the next chunk, considering overlap
        start = end - overlap if end < text_length else text_length

    return chunks

def extract_key_concepts(text):
    """
    Extract key concepts from text using simple NLP techniques.
    This helps focus the question generation on important topics.
    """
    # Simple implementation: extract sentences with important keywords
    important_keywords = [
        'important', 'significant', 'key', 'main', 'critical', 'essential',
        'fundamental', 'crucial', 'vital', 'primary', 'major', 'central',
        'define', 'definition', 'concept', 'theory', 'principle', 'method',
        'process', 'system', 'function', 'structure', 'example'
    ]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    key_sentences = []

    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence.lower())
        if any(keyword in words for keyword in important_keywords):
            key_sentences.append(sentence)

    # If we found less than 5 key sentences, add more sentences until we have at least 5
    # or until we've used all sentences
    if len(key_sentences) < 5 and len(sentences) > 5:
        remaining_sentences = [s for s in sentences if s not in key_sentences]
        key_sentences.extend(remaining_sentences[:5 - len(key_sentences)])

    return ' '.join(key_sentences)

def create_question_templates(num_questions, difficulty):
    """
    Create question templates locally based on difficulty level.
    This reduces the work the AI needs to do.
    """
    templates = []

    # Define question types based on difficulty
    if difficulty == 'easy':
        question_types = [
            "What is the definition of {concept}?",
            "Which of the following best describes {concept}?",
            "What is the main purpose of {concept}?",
            "In what context would you use {concept}?",
            "Which example demonstrates {concept}?"
        ]
    elif difficulty == 'medium':
        question_types = [
            "How does {concept} relate to {other_concept}?",
            "What are the implications of {concept}?",
            "Compare and contrast {concept} and {other_concept}.",
            "What is the significance of {concept} in the context of {topic}?",
            "How would you apply {concept} to solve {problem}?"
        ]
    else:  # hard
        question_types = [
            "Analyze the relationship between {concept} and {other_concept} in the context of {topic}.",
            "Evaluate the effectiveness of {concept} for {purpose}.",
            "What would happen if {concept} was applied differently in {scenario}?",
            "Critique the use of {concept} in {context}.",
            "Synthesize a new approach using {concept} and {other_concept}."
        ]

    # Create templates by cycling through question types
    for i in range(num_questions):
        templates.append(question_types[i % len(question_types)])

    return templates

def parse_quiz_content(content):
    """Parse the AI-generated quiz content into structured data."""
    questions = []

    # Split content into individual questions
    # This regex pattern looks for numbered questions (1., 2., etc.)
    question_blocks = re.split(r'\n\s*\d+\.\s+', content)

    # Remove any empty blocks at the beginning
    if question_blocks and not question_blocks[0].strip():
        question_blocks = question_blocks[1:]

    for block in question_blocks:
        if not block.strip():
            continue

        # Extract question text and options
        lines = block.strip().split('\n')
        question_text = lines[0].strip()

        options = {}
        correct_option = None
        topic = None

        # Look for options (a), b), c), d) or A., B., C., D.)
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # Try to extract option
            option_match = re.match(r'^([a-dA-D])[\.\)]\s+(.+)$', line)
            if option_match:
                option_letter = option_match.group(1).lower()
                option_text = option_match.group(2).strip()

                # Check if this option is marked as correct
                if '(correct)' in option_text.lower():
                    correct_option = option_letter
                    option_text = re.sub(r'\s*\(correct\)\s*', '', option_text, flags=re.IGNORECASE)

                options[option_letter] = option_text

            # Try to extract correct answer if it's specified separately
            correct_match = re.match(r'^correct\s+answer\s*:\s*([a-dA-D])', line, re.IGNORECASE)
            if correct_match:
                correct_option = correct_match.group(1).lower()

            # Try to extract topic if it's specified
            topic_match = re.match(r'^topic\s*:\s*(.+)$', line, re.IGNORECASE)
            if topic_match:
                topic = topic_match.group(1).strip()

        # Only add well-formed questions with all options and a correct answer
        if question_text and len(options) == 4 and correct_option:
            questions.append({
                'text': question_text,
                'options': {
                    'a': options.get('a', ''),
                    'b': options.get('b', ''),
                    'c': options.get('c', ''),
                    'd': options.get('d', '')
                },
                'correct_option': correct_option,
                'topic': topic
            })

    return questions

def generate_quiz_from_pdf(pdf_file, num_questions, difficulty):
    # Save the PDF file first to enable caching
    pdf_obj = UploadedPDF.objects.create(file=pdf_file)

    # Extract text with caching
    full_text = extract_text_from_pdf(pdf_file, pdf_obj)

    # Process text locally to minimize API usage
    chunks = chunk_text(full_text, max_chunk_size=2000, overlap=200)

    # Extract key concepts from each chunk and combine
    processed_content = ""
    for chunk in chunks:
        key_concepts = extract_key_concepts(chunk)
        processed_content += key_concepts + " "

    # Trim to a reasonable size, focusing on the beginning which often contains
    # important introductory content, and the extracted key concepts
    if len(processed_content) > 3000:
        # Take the first 1000 characters (likely introduction)
        intro = full_text[:1000]
        # Take the most important processed content to fill the remaining space
        remaining_space = 2000
        key_content = processed_content[:remaining_space]
        optimized_content = intro + "\n\n" + key_content
    else:
        optimized_content = processed_content

    # Create question templates locally
    templates = create_question_templates(num_questions, difficulty)
    template_examples = "\n".join(templates[:3])  # Send just a few examples to save tokens

    # Initialize the LLM with a lower temperature for more focused responses
    llm = ChatOpenAI(temperature=0.5, openai_api_key=settings.OPENAI_API_KEY)

    # Create an optimized prompt that leverages our local processing
    prompt = ChatPromptTemplate.from_template("""
    You are a precise quiz generator focusing on educational assessment.

    I've already extracted key concepts and created question templates for you.
    Generate {n} {difficulty} multiple-choice questions based on these key concepts:

    {content}

    Use these question template examples as inspiration:
    {templates}

    For each question:
    1. Focus on testing understanding, not just memorization
    2. Include 4 options labeled a), b), c), and d)
    3. Mark the correct option with (correct) at the end
    4. Include a topic label that categorizes the concept being tested

    Format each question exactly like this:
    1. [Question text]
    a) [Option text]
    b) [Option text]
    c) [Option text] (correct)
    d) [Option text]
    Topic: [Topic name]

    Be concise and precise in your language.
    """)

    final_prompt = prompt.format_messages(
        content=optimized_content, 
        n=num_questions, 
        difficulty=difficulty,
        templates=template_examples
    )

    # Track token usage
    with get_openai_callback() as cb:
        response = llm(final_prompt)

        # Save token usage to database
        token_usage = TokenUsage.objects.create(
            prompt_tokens=cb.prompt_tokens,
            completion_tokens=cb.completion_tokens,
            total_tokens=cb.total_tokens
        )

        # Parse the quiz content into structured data
        parsed_questions = parse_quiz_content(response.content)

        # Create the quiz
        quiz = Quiz.objects.create(
            pdf=pdf_obj,
            num_questions=num_questions,
            difficulty=difficulty
        )

        # Create the questions
        for q_data in parsed_questions:
            Question.objects.create(
                quiz=quiz,
                text=q_data['text'],
                option_a=q_data['options']['a'],
                option_b=q_data['options']['b'],
                option_c=q_data['options']['c'],
                option_d=q_data['options']['d'],
                correct_option=q_data['correct_option'],
                topic=q_data['topic']
            )

        # Return both the quiz object, raw content, and token usage
        return {
            'quiz': quiz,
            'content': response.content,
            'token_usage': token_usage
        }
