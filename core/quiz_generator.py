import io
import re
import json
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
import mmap
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader, PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from django.conf import settings
from django.core.cache import cache
from .models import TokenUsage, UploadedPDF, Quiz, Question
import gc
import tempfile
import pickle
import lzma

# Global thread-safe cache for text chunks
_chunk_cache = {}
_chunk_cache_lock = threading.Lock()


def ensure_text_chunks_directory(pdf_id):
    """Fast directory creation with minimal error handling."""
    chunks_dir = f"/app/media/text_chunks/{pdf_id}"
    os.makedirs(chunks_dir, exist_ok=True)
    return chunks_dir


def extract_text_from_pdf(pdf_file, pdf_obj=None):
    """
    Extract text from PDF with multiple fallback methods and aggressive caching.
    """
    return extract_text_from_pdf_fast(pdf_file, pdf_obj)

def extract_text_from_pdf_fast(pdf_file, pdf_obj=None):
    """
    Optimized text extraction with multiple fallback methods and aggressive caching.
    """
    # Check cache first
    if pdf_obj:
        cache_key = f"pdf_text_{pdf_obj.id}"
        cached_text = cache.get(cache_key)
        if cached_text:
            return cached_text

        if pdf_obj.extracted_text:
            cache.set(cache_key, pdf_obj.extracted_text, timeout=86400)
            return pdf_obj.extracted_text

    # Use the fastest PDF extraction method
    try:
        # Method 1: Try PyPDF2 first (often fastest for simple PDFs)
        if hasattr(pdf_file, 'read'):
            pdf_file.seek(0)
            pdf_data = pdf_file.read()
            pdf_file.seek(0)
        else:
            with open(pdf_file, 'rb') as f:
                pdf_data = f.read()

        # Use BytesIO for in-memory processing
        pdf_stream = io.BytesIO(pdf_data)
        reader = PdfReader(pdf_stream)

        # Parallel page extraction
        def extract_page_text(page_num):
            try:
                return reader.pages[page_num].extract_text()
            except:
                return ""

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(4, len(reader.pages))) as executor:
            page_texts = list(executor.map(extract_page_text, range(len(reader.pages))))

        text = " ".join(page_texts)

        # If PyPDF2 fails or returns empty, fallback to PyPDFium2
        if not text.strip():
            raise Exception("PyPDF2 returned empty text")

    except Exception as e:
        print(f"PyPDF2 failed: {e}, trying PyPDFium2")
        try:
            # Fallback method with temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                if hasattr(pdf_file, 'read'):
                    pdf_file.seek(0)
                    temp_file.write(pdf_file.read())
                    pdf_file.seek(0)
                else:
                    with open(pdf_file, 'rb') as f:
                        temp_file.write(f.read())
                temp_path = temp_file.name

            # Use PyPDFium2 with minimal splitting
            loader = PyPDFium2Loader(temp_path)
            pages = loader.load()
            text = " ".join([page.page_content for page in pages])

            # Cleanup
            os.unlink(temp_path)
        except Exception as e2:
            print(f"PyPDFium2 also failed: {e2}")
            text = ""

    # Cache the result
    if pdf_obj and text:
        pdf_obj.save_extracted_text(text)
        cache.set(cache_key, text, timeout=86400)

    return text


def chunk_text_fast(text, max_chunk_size=1000, overlap=200):
    """
    Ultra-fast text chunking using string operations instead of regex.
    """
    if not text:
        return []

    chunks = []
    text_length = len(text)
    start = 0

    # Pre-compile break patterns for speed
    break_patterns = ['\n\n', '. ', '! ', '? ', '\n', ' ']

    while start < text_length:
        end = min(start + max_chunk_size, text_length)

        # Find the best break point
        if end < text_length:
            best_break = -1
            for pattern in break_patterns:
                break_pos = text.rfind(pattern, start + max_chunk_size // 2, end)
                if break_pos > best_break:
                    best_break = break_pos + len(pattern)
                    break

            if best_break > start:
                end = best_break

        chunks.append(text[start:end])
        start = max(start + 1, end - overlap)

    return chunks


def store_text_chunks_fast(text_chunks, pdf_id, task_id=None):
    """
    Optimized chunk storage using compressed files and batch operations.
    """
    cache_key = f"text_chunks_{pdf_id}"
    if cache.get(cache_key):
        return ensure_text_chunks_directory(pdf_id)

    chunks_dir = ensure_text_chunks_directory(pdf_id)

    # Store all chunks in a single compressed file
    chunks_file = os.path.join(chunks_dir, "chunks.pkl.xz")

    try:
        # Convert generator to list if needed
        if hasattr(text_chunks, '__iter__') and not isinstance(text_chunks, list):
            text_chunks = list(text_chunks)

        # Compress and store chunks
        with lzma.open(chunks_file, 'wb') as f:
            pickle.dump(text_chunks, f)

        # Cache the reference
        cache.set(cache_key, True, timeout=86400)

        if task_id:
            from .tasks import update_progress
            update_progress(task_id, 3, 5, "Text chunks stored successfully")

    except Exception as e:
        print(f"Error storing chunks: {e}")
        # Fallback to individual files
        for i, chunk in enumerate(text_chunks):
            with open(os.path.join(chunks_dir, f"chunk_{i}.txt"), 'w') as f:
                f.write(chunk)
        cache.set(cache_key, True, timeout=86400)

    return chunks_dir


def retrieve_relevant_chunks_fast(query, pdf_id, num_chunks=5):
    """
    Optimized chunk retrieval using compressed storage and efficient search.
    """
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cache_key = f"query_results_{pdf_id}_{query_hash}_{num_chunks}"

    cached_results = cache.get(cache_key)
    if cached_results:
        return cached_results

    chunks_dir = ensure_text_chunks_directory(pdf_id)
    chunks_file = os.path.join(chunks_dir, "chunks.pkl.xz")

    try:
        # Load compressed chunks
        with lzma.open(chunks_file, 'rb') as f:
            all_chunks = pickle.load(f)
    except:
        # Fallback to individual files
        all_chunks = []
        chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.startswith('chunk_')])
        for chunk_file in chunk_files:
            with open(os.path.join(chunks_dir, chunk_file), 'r') as f:
                all_chunks.append(f.read())

    # Fast keyword extraction and scoring
    query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
    query_words -= {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our',
                    'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
                    'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}

    # Score chunks in parallel
    def score_chunk(chunk):
        chunk_lower = chunk.lower()
        score = sum(chunk_lower.count(word) for word in query_words)
        return (chunk, score)

    # Use threading for faster processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        scored_chunks = list(executor.map(score_chunk, all_chunks))

    # Sort and filter
    scored_chunks = [(chunk, score) for chunk, score in scored_chunks if score > 0]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Get top chunks
    relevant_chunks = scored_chunks[:num_chunks]

    # If not enough, add random ones
    if len(relevant_chunks) < num_chunks:
        remaining = [(chunk, 0) for chunk, score in scored_chunks[num_chunks:]]
        relevant_chunks.extend(remaining[:num_chunks - len(relevant_chunks)])

    # Cache results
    cache.set(cache_key, relevant_chunks, timeout=3600)
    return relevant_chunks


@lru_cache(maxsize=128)
def create_question_templates_cached(num_questions, difficulty):
    """Cached question template creation."""
    templates = {
        'easy': [
            "What is {concept}?",
            "Which best describes {concept}?",
            "What is the main purpose of {concept}?",
            "When would you use {concept}?",
            "Which example shows {concept}?"
        ],
        'medium': [
            "How does {concept} relate to {other_concept}?",
            "What are the implications of {concept}?",
            "Compare {concept} and {other_concept}.",
            "What is the significance of {concept}?",
            "How would you apply {concept}?"
        ],
        'hard': [
            "Analyze the relationship between {concept} and {other_concept}.",
            "Evaluate the effectiveness of {concept}.",
            "What would happen if {concept} was modified?",
            "Critique the use of {concept}.",
            "Synthesize a new approach using {concept}."
        ]
    }

    question_types = templates.get(difficulty, templates['medium'])
    return [question_types[i % len(question_types)] for i in range(num_questions)]


def parse_quiz_content_fast(content):
    """Optimized quiz parsing with pre-compiled regex."""
    questions = []

    # Pre-compile regex patterns
    question_split_pattern = re.compile(r'\n\s*\d+\.\s+')
    option_pattern = re.compile(r'^([a-dA-D])[\.\)]\s+(.+)$')
    correct_pattern = re.compile(r'^correct\s+answer\s*:\s*([a-dA-D])', re.IGNORECASE)
    topic_pattern = re.compile(r'^topic\s*:\s*(.+)$', re.IGNORECASE)

    # Split into questions
    question_blocks = question_split_pattern.split(content)
    if question_blocks and not question_blocks[0].strip():
        question_blocks = question_blocks[1:]

    for block in question_blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')
        question_text = lines[0].strip()

        options = {}
        correct_option = None
        topic = None

        # Process lines efficiently
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # Check for option
            option_match = option_pattern.match(line)
            if option_match:
                option_letter = option_match.group(1).lower()
                option_text = option_match.group(2).strip()

                if '(correct)' in option_text.lower():
                    correct_option = option_letter
                    option_text = re.sub(r'\s*\(correct\)\s*', '', option_text, flags=re.IGNORECASE)

                options[option_letter] = option_text
                continue

            # Check for correct answer
            correct_match = correct_pattern.match(line)
            if correct_match:
                correct_option = correct_match.group(1).lower()
                continue

            # Check for topic
            topic_match = topic_pattern.match(line)
            if topic_match:
                topic = topic_match.group(1).strip()

        # Add well-formed questions
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


def generate_quiz_from_pdf(pdf_obj, num_questions, difficulty, task_id=None):
    """
    Ultra-fast quiz generation with optimized processing pipeline.
    """

    def update_progress_fast(step, message):
        if task_id:
            try:
                from .tasks import update_progress
                update_progress(task_id, step, 5, message)
            except:
                pass

    try:
        # Step 1: Extract text (with caching)
        update_progress_fast(1, "Extracting text from PDF...")
        full_text = extract_text_from_pdf_fast(pdf_obj.file, pdf_obj)

        if not full_text.strip():
            raise Exception("No text could be extracted from the PDF")

        # Step 2: Smart chunking based on text size
        update_progress_fast(2, "Processing text...")
        text_length = len(full_text)

        if text_length > 500000:  # Large PDF
            chunk_size = 2000
            overlap = 300
            num_chunks_to_use = min(8, num_questions * 2)
        else:  # Normal PDF
            chunk_size = 1000
            overlap = 200
            num_chunks_to_use = min(5, num_questions)

        # Fast chunking
        chunks = chunk_text_fast(full_text, chunk_size, overlap)

        # Step 3: Store chunks efficiently
        update_progress_fast(3, "Storing chunks...")
        chunks_dir = store_text_chunks_fast(chunks, pdf_obj.id, task_id)

        # Step 4: Find relevant content
        update_progress_fast(4, "Finding relevant content...")
        query = f"Generate {num_questions} {difficulty} questions main topics concepts"
        relevant_chunks = retrieve_relevant_chunks_fast(query, pdf_obj.id, num_chunks_to_use)

        # Prepare content for AI
        content_parts = [chunk for chunk, score in relevant_chunks if score > 0]
        if not content_parts:
            content_parts = [chunk for chunk, score in relevant_chunks[:3]]

        # Limit content size for faster processing
        optimized_content = "\n\n".join(content_parts)
        if len(optimized_content) > 4000:
            optimized_content = optimized_content[:4000]

        # Step 5: Generate quiz with AI
        update_progress_fast(5, "Generating quiz with AI...")

        # Use faster LLM settings
        llm = ChatOpenAI(
            temperature=0.3,  # Lower temperature for faster, more focused responses
            max_tokens=1500,  # Limit response length
            model_name="gpt-3.5-turbo"  # Use faster model
        )

        # Streamlined prompt
        prompt = ChatPromptTemplate.from_template("""
Generate {n} {difficulty} multiple-choice questions from this content:

{content}

Format each question as:
1. [Question]
a) [Option]
b) [Option]
c) [Option] (correct)
d) [Option]

Be concise and test key concepts.
""")

        # Generate quiz
        with get_openai_callback() as cb:
            response = llm(prompt.format_messages(
                content=optimized_content,
                n=num_questions,
                difficulty=difficulty
            ))

            # Save token usage
            token_usage = TokenUsage.objects.create(
                prompt_tokens=cb.prompt_tokens,
                completion_tokens=cb.completion_tokens,
                total_tokens=cb.total_tokens
            )

        # Parse and save quiz
        parsed_questions = parse_quiz_content_fast(response.content)

        # Create quiz object
        quiz = Quiz.objects.create(
            pdf=pdf_obj,
            num_questions=num_questions,
            difficulty=difficulty
        )

        # Bulk create questions for better performance
        question_objects = []
        for q_data in parsed_questions:
            question_objects.append(Question(
                quiz=quiz,
                text=q_data['text'],
                option_a=q_data['options']['a'],
                option_b=q_data['options']['b'],
                option_c=q_data['options']['c'],
                option_d=q_data['options']['d'],
                correct_option=q_data['correct_option'],
                topic=q_data['topic']
            ))

        # Bulk insert
        Question.objects.bulk_create(question_objects)

        # Cache result
        cache_key = f"quiz_result_{quiz.id}"
        cache_data = json.dumps([quiz.id, response.content, token_usage.id])
        cache.set(cache_key, cache_data, timeout=86400)

        return {
            'quiz': quiz,
            'content': response.content,
            'token_usage': token_usage
        }

    except Exception as e:
        print(f"Error in quiz generation: {e}")
        raise
