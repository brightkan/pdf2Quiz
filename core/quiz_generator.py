import io
import re
import json
import hashlib
import os
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from django.conf import settings
from django.core.cache import cache
from .models import TokenUsage, UploadedPDF, Quiz, Question
import gc
import tempfile

def ensure_text_chunks_directory(pdf_id):
    """
    Ensure the directory for storing text chunks exists.

    Args:
        pdf_id: The ID of the PDF

    Returns:
        str: The path to the chunks directory
    """
    # Use persistent directory that's accessible in Docker
    chunks_dir = f"/app/media/text_chunks/{pdf_id}"
    print(f"Using chunks directory: {chunks_dir}")

    # Ensure the directory exists
    try:
        os.makedirs(chunks_dir, exist_ok=True)
        print(f"Ensured directory exists: {chunks_dir}")
    except Exception as e:
        print(f"Error creating directory {chunks_dir}: {e}")
        # Fall back to /tmp if we can't create the directory
        chunks_dir = f"/tmp/text_chunks/{pdf_id}"
        print(f"Falling back to {chunks_dir}")
        os.makedirs(chunks_dir, exist_ok=True)

    return chunks_dir

def extract_text_from_pdf(pdf_file, pdf_obj=None):
    """
    Extract text from a PDF file and cache it using Redis if a pdf_obj is provided.
    If the text is already cached, return it instead of re-extracting.
    Uses LangChain's PyPDFLoader and RecursiveCharacterTextSplitter for efficient processing of large PDFs.
    """
    print(f"Starting extract_text_from_pdf: pdf_obj_id={pdf_obj.id if pdf_obj else 'None'}")

    # Generate a cache key based on the PDF file name or object ID
    cache_key = None
    if pdf_obj:
        cache_key = f"pdf_text_{pdf_obj.id}"
        print(f"Cache key: {cache_key}")

        # Try to get the text from Redis cache first
        cached_text = cache.get(cache_key)
        if cached_text:
            print(f"Found cached text in Redis for pdf_id={pdf_obj.id}")
            return cached_text

        # If not in Redis but in database, get it and cache it
        if pdf_obj.extracted_text:
            print(f"Found extracted text in database for pdf_id={pdf_obj.id}")
            cache.set(cache_key, pdf_obj.extracted_text, timeout=86400)  # Cache for 24 hours
            return pdf_obj.extracted_text

    # Otherwise extract the text using LangChain's PyPDFLoader
    try:
        print("Extracting text from PDF file using LangChain's PyPDFLoader")

        # Save the file to a temporary location if it's an in-memory file
        if hasattr(pdf_file, 'temporary_file_path'):
            # Django's UploadedFile already has a path
            pdf_path = pdf_file.temporary_file_path()
        else:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_file.read())
                pdf_path = temp_file.name
                # Reset file pointer for potential future use
                if hasattr(pdf_file, 'seek'):
                    pdf_file.seek(0)

        # Use PyPDFium2Loader to load the PDF (faster)
        loader = PyPDFium2Loader(pdf_path)

        # Create a text splitter for efficient processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Load and split the document in one step
        pages = loader.load_and_split(text_splitter)

        # Process pages in batches to reduce memory usage
        all_text = []
        batch_size = 20  # Process 20 pages at a time

        for i in range(0, len(pages), batch_size):
            batch = pages[i:i+batch_size]
            batch_text = [page.page_content for page in batch]
            all_text.extend(batch_text)

            # Free up memory after processing each batch
            batch = None
            batch_text = None
            if i % (batch_size * 5) == 0:  # Every 5 batches
                gc.collect()

        # Join text only when needed
        text = " ".join(all_text)
        print(f"Extracted {len(text)} characters of text using PyPDFium2Loader")

        # Clean up temporary file if we created one
        if not hasattr(pdf_file, 'temporary_file_path') and os.path.exists(pdf_path):
            os.unlink(pdf_path)

        # Clear the list to free memory
        all_text = None
        pages = None
        gc.collect()

    except Exception as e:
        print(f"Error extracting text with PyPDFium2Loader: {e}")
        import traceback
        traceback.print_exc()
        # Return an empty string if extraction fails
        text = ""

    # Cache the extracted text in Redis and database if pdf_obj is provided
    if pdf_obj and text:
        try:
            print(f"Saving extracted text to database for pdf_id={pdf_obj.id}")
            pdf_obj.save_extracted_text(text)
            print(f"Caching extracted text in Redis for pdf_id={pdf_obj.id}")
            cache.set(cache_key, text, timeout=86400)  # Cache for 24 hours
        except Exception as save_error:
            print(f"Error saving or caching extracted text: {save_error}")
            import traceback
            traceback.print_exc()

    return text

def chunk_text(text, max_chunk_size=500, overlap=100):
    """
    Split text into chunks of maximum size with overlap between chunks.
    This helps maintain context between chunks.
    Uses a generator approach to reduce memory usage.
    Optimized for very large texts with smarter boundary detection.
    """
    start = 0
    text_length = len(text)

    # For very large texts, use a more efficient approach
    is_very_large = text_length > 5000000  # > 5MB of text

    # Use a sliding window approach for efficiency
    while start < text_length:
        # Determine the end position for this chunk
        end = min(start + max_chunk_size, text_length)

        # If we're not at the end of the text, try to find a good breaking point
        if end < text_length:
            # For very large texts, limit the search range to improve performance
            search_start = end - min(200, end - start) if is_very_large else start

            # Try to find the last paragraph break first (most natural boundary)
            last_para = text.rfind('\n\n', search_start, end)

            # If we found a paragraph break, use it
            if last_para != -1 and last_para > start + (max_chunk_size // 2):
                # Only use paragraph breaks that give us a reasonable chunk size
                # (at least half of max_chunk_size)
                end = last_para + 2  # Include the newline characters
            else:
                # Try to find the last sentence boundary (period, question mark, exclamation point)
                last_period = max(
                    text.rfind('. ', search_start, end),
                    text.rfind('? ', search_start, end),
                    text.rfind('! ', search_start, end)
                )

                # If we found a sentence boundary, use it
                if last_period != -1 and last_period > start + (max_chunk_size // 4):
                    # Only use sentence breaks that give us a reasonable chunk size
                    # (at least quarter of max_chunk_size)
                    end = last_period + 2  # Include the period and space
                elif not is_very_large:
                    # For smaller texts, try to find the last space for word boundary
                    # Skip this for very large texts as it's less critical and more expensive
                    last_space = text.rfind(' ', search_start, end)
                    if last_space != -1 and last_space > start + (max_chunk_size // 8):
                        end = last_space + 1  # Include the space

        # Yield the chunk
        yield text[start:end]

        # Move the start position for the next chunk, considering overlap
        start = end - overlap if end < text_length else text_length

        # Run garbage collection periodically, but less frequently for very large texts
        gc_interval = max_chunk_size * (50 if is_very_large else 10)
        if start % gc_interval < max_chunk_size:
            gc.collect()

def get_text_chunks(text, max_chunk_size=500, overlap=100):
    """
    Wrapper function to get all chunks as a list when needed.
    For most operations, use chunk_text directly as a generator.
    """
    return list(chunk_text(text, max_chunk_size, overlap))

def store_text_chunks(text_chunks, pdf_id, task_id=None):
    """
    Store text chunks in files for later retrieval.
    Uses batch processing and streaming for better memory efficiency.
    Optimized for very large PDFs with adaptive batch sizing and compression.

    Performance optimizations:
    - Increased batch size for normal PDFs (20 chunks per batch)
    - Reduced progress reporting frequency to minimize Redis overhead
    - Optimized file writing with single write operations
    - Reduced garbage collection frequency to improve performance
    - Improved progress reporting accuracy with dynamic scaling

    Args:
        text_chunks: Either a list of chunks or a generator that yields chunks
        pdf_id: The ID of the PDF
        task_id: Optional Celery task ID for progress tracking

    Returns:
        str: The path to the directory where chunks are stored
    """
    # Generate a unique directory name based on the PDF ID
    print(f"store_text_chunks: pdf_id={pdf_id}")

    # Check if we already have cached chunks
    cache_key = f"text_chunks_{pdf_id}"
    cached_exists = cache.get(cache_key)
    print(f"Cached chunks exist: {cached_exists}")

    # Ensure the directory exists
    chunks_dir = ensure_text_chunks_directory(pdf_id)

    if not cached_exists:
        # Estimate if we're dealing with a very large PDF based on directory size
        # or other heuristics if available
        try:
            is_very_large = False
            if os.path.exists(chunks_dir):
                # Check if there are already some chunks that might indicate size
                existing_files = [f for f in os.listdir(chunks_dir) if f.startswith('batch_')]
                if existing_files:
                    # Sample the first file to estimate size
                    sample_path = os.path.join(chunks_dir, existing_files[0])
                    sample_size = os.path.getsize(sample_path)
                    # If the sample is large, assume we're dealing with a large PDF
                    is_very_large = sample_size > 1000000  # > 1MB per batch file
        except Exception as e:
            print(f"Error estimating PDF size: {e}")
            is_very_large = False

        # Adaptive batch size based on estimated PDF size
        # Use smaller batches for very large PDFs to reduce memory pressure
        # but larger batches for normal PDFs to improve performance
        batch_size = 5 if is_very_large else 20
        print(f"Using batch size {batch_size} for {'very large' if is_very_large else 'normal'} PDF")

        # Process chunks in a streaming fashion
        batch_texts = []
        chunk_count = 0
        batch_count = 0
        total_processed = 0

        # Estimate total chunks for progress reporting (if text_chunks is a generator)
        try:
            if hasattr(text_chunks, '__len__'):
                total_chunks = len(text_chunks)
                is_generator = False
            else:
                # Better estimate for very large PDFs
                if is_very_large:
                    total_chunks = 500  # Assume more chunks for large PDFs
                else:
                    total_chunks = 100  # Default estimate
                is_generator = True
        except:
            total_chunks = 100  # Default estimate
            is_generator = True

        print(f"Processing chunks as {'generator' if is_generator else 'list'}, estimated total: {total_chunks}")

        # For very large PDFs, use a more efficient progress reporting approach
        # Increase the interval for normal PDFs to reduce overhead
        progress_interval = 10 if is_very_large else 3
        last_progress_update = 0

        # Process chunks one by one
        for chunk in text_chunks:
            batch_texts.append(chunk)
            chunk_count += 1

            # When we have enough chunks, process them as a batch
            if len(batch_texts) >= batch_size:
                batch_count += 1
                try:
                    # Save the batch to a file
                    batch_file = os.path.join(chunks_dir, f"batch_{batch_count}.txt")

                    # Use a more efficient file writing approach with a single write operation
                    with open(batch_file, 'w', encoding='utf-8') as f:
                        # Prepare the content as a single string to minimize I/O operations
                        content = []
                        for i, text in enumerate(batch_texts):
                            chunk_id = total_processed + i + 1
                            content.append(f"# chunk_{chunk_id}\n{text}\n\n---\n\n")

                        # Write all content at once
                        f.write(''.join(content))

                    total_processed += len(batch_texts)

                    # Progress update (less frequent for very large PDFs)
                    if task_id and (batch_count - last_progress_update >= progress_interval):
                        from .tasks import update_progress
                        last_progress_update = batch_count

                        if is_generator:
                            # For generators, use a more accurate progress estimation
                            # Calculate progress based on batches processed and adjust the scale
                            # to ensure we reach 100% at completion
                            batch_progress = (batch_count * batch_size)
                            # Use a dynamic scaling factor that increases as we process more batches
                            scaling_factor = 1.0 + (batch_count / 50.0)  # Increases as we process more
                            estimated_progress = batch_progress * scaling_factor
                            percent = min(int((estimated_progress / total_chunks) * 100), 99)

                            # If we've processed a significant number of batches, increase the percent
                            # This helps prevent the progress from appearing stuck
                            if batch_count > 20:
                                percent = max(percent, 80)  # Ensure we're at least at 80% after 20 batches
                            if batch_count > 50:
                                percent = max(percent, 90)  # Ensure we're at least at 90% after 50 batches
                            if batch_count > 100:
                                percent = max(percent, 95)  # Ensure we're at least at 95% after 100 batches
                        else:
                            # For lists, we know the exact progress
                            percent = int((total_processed / total_chunks) * 100)

                        update_progress(task_id, 3, 5, 
                                       f"Processing text chunks... batch {batch_count} ({percent}%)")

                except MemoryError as mem_err:
                    print(f"MemoryError during chunk processing batch {batch_count}: {mem_err}")
                    if task_id:
                        from .tasks import update_progress
                        update_progress(task_id, 3, 5, f"Failed: Out of memory during chunk processing batch {batch_count}")

                    # For memory errors, try to recover by reducing batch size
                    if len(batch_texts) > 1:
                        # Save half the batch and retry with smaller batches
                        mid_point = len(batch_texts) // 2
                        try:
                            # Save first half with optimized writing
                            recovery_file = os.path.join(chunks_dir, f"recovery_{batch_count}_1.txt")
                            with open(recovery_file, 'w', encoding='utf-8') as f:
                                content = []
                                for i, text in enumerate(batch_texts[:mid_point]):
                                    chunk_id = total_processed + i + 1
                                    content.append(f"# chunk_{chunk_id}\n{text}\n\n---\n\n")
                                f.write(''.join(content))

                            # Save second half with optimized writing
                            recovery_file = os.path.join(chunks_dir, f"recovery_{batch_count}_2.txt")
                            with open(recovery_file, 'w', encoding='utf-8') as f:
                                content = []
                                for i, text in enumerate(batch_texts[mid_point:]):
                                    chunk_id = total_processed + mid_point + i + 1
                                    content.append(f"# chunk_{chunk_id}\n{text}\n\n---\n\n")
                                f.write(''.join(content))

                            total_processed += len(batch_texts)
                            print(f"Recovered from memory error by splitting batch {batch_count}")

                            # Reduce batch size for future batches
                            batch_size = max(1, batch_size // 2)
                            print(f"Reduced batch size to {batch_size}")

                        except Exception as recovery_err:
                            print(f"Failed to recover from memory error: {recovery_err}")
                            raise mem_err
                    else:
                        # Can't reduce batch size further
                        raise

                except Exception as e:
                    print(f"Error during chunk processing batch {batch_count}: {e}")
                    if task_id:
                        from .tasks import update_progress
                        update_progress(task_id, 3, 5, f"Failed: Error during chunk processing batch {batch_count}")
                    raise

                # Clear the batch and run garbage collection
                batch_texts = []

                # Run garbage collection less frequently to reduce overhead
                # For very large PDFs, run every 10 batches
                # For normal PDFs, run every 5 batches
                if batch_count % (10 if is_very_large else 5) == 0:
                    gc.collect()

        # Process any remaining chunks
        if batch_texts:
            batch_count += 1
            try:
                # Save the remaining chunks to a file with optimized writing
                batch_file = os.path.join(chunks_dir, f"batch_{batch_count}.txt")
                with open(batch_file, 'w', encoding='utf-8') as f:
                    # Prepare the content as a single string to minimize I/O operations
                    content = []
                    for i, text in enumerate(batch_texts):
                        # Add metadata as a comment line
                        chunk_id = total_processed + i + 1
                        content.append(f"# chunk_{chunk_id}\n{text}\n\n---\n\n")

                    # Write all content at once
                    f.write(''.join(content))

                # Final progress update - ensure we show 100% completion
                if task_id:
                    from .tasks import update_progress
                    # Force progress to 100% for this step to indicate completion
                    update_progress(task_id, 3, 5, f"Processing text chunks... complete (100%)")

                    # Add a small delay to ensure the progress update is visible to the user
                    import time
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error during final chunk processing batch: {e}")
                if task_id:
                    from .tasks import update_progress
                    update_progress(task_id, 3, 5, f"Failed: Error during final chunk processing batch")
                raise

        # Cache the chunks reference
        cache.set(cache_key, True, timeout=86400)  # Cache for 24 hours

        # Final garbage collection
        batch_texts = None
        gc.collect()

    return chunks_dir

def retrieve_relevant_chunks(query, pdf_id, num_chunks=5):
    """
    Retrieve the most relevant chunks for a given query using keyword matching.
    Uses caching to improve performance for repeated queries.
    Optimized for memory efficiency.
    Returns a list of (chunk_text, score) tuples.
    """
    print(f"retrieve_relevant_chunks: query={query}, pdf_id={pdf_id}, num_chunks={num_chunks}")

    # Create a cache key for this specific query
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cache_key = f"query_results_{pdf_id}_{query_hash}_{num_chunks}"
    print(f"Cache key: {cache_key}")

    # Try to get cached results first
    cached_results = cache.get(cache_key)
    if cached_results:
        print(f"Found cached results for query")
        return cached_results

    print(f"No cached results found, searching text chunks")

    # Get the directory where chunks are stored
    chunks_dir = ensure_text_chunks_directory(pdf_id)

    # Extract keywords from the query
    # Remove common words and keep only meaningful terms
    common_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'about', 'from', 'of'}
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    keywords = [word for word in query_words if word not in common_words and len(word) > 2]

    print(f"Extracted keywords from query: {keywords}")

    # Initialize variables
    all_scored_chunks = [] # List to store (chunk_text, score) tuples

    try:
        # Get all batch files in the chunks directory
        batch_files = [f for f in os.listdir(chunks_dir) if f.startswith('batch_') and f.endswith('.txt')]
        print(f"Found {len(batch_files)} batch files")

        # Process each batch file
        for batch_file in batch_files:
            file_path = os.path.join(chunks_dir, batch_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split the content into individual chunks using the separator
            chunks = content.split("\n\n---\n\n")

            # Process each chunk
            for chunk in chunks:
                if not chunk.strip():
                    continue

                # Remove metadata line if present
                chunk_lines = chunk.split('\n')
                if chunk_lines[0].startswith('#'):
                    chunk_text = '\n'.join(chunk_lines[1:])
                else:
                    chunk_text = chunk

                # Skip empty chunks
                if not chunk_text.strip():
                    continue

                # Calculate relevance score based on keyword matches
                score = 0
                chunk_lower = chunk_text.lower()

                # Count keyword occurrences
                for keyword in keywords:
                    # Count exact matches
                    exact_matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', chunk_lower))
                    score += exact_matches * 2  # Weight exact matches more

                    # Count partial matches
                    if len(keyword) > 4:  # Only consider partial matches for longer keywords
                        partial_matches = chunk_lower.count(keyword) - exact_matches
                        score += partial_matches

                # Store the score
                all_scored_chunks.append((chunk_text, score))

        # Sort chunks by relevance score (descending)
        sorted_chunks = sorted(all_scored_chunks, key=lambda x: x[1], reverse=True)

        # Get the top N chunks with a score greater than 0
        relevant_chunks_with_scores = [(chunk, score) for chunk, score in sorted_chunks if score > 0][:num_chunks]

        # If we don't have enough relevant chunks, add random ones to meet the requested number
        if len(relevant_chunks_with_scores) < num_chunks and all_scored_chunks:
            # Get chunks that aren't already in relevant_chunks_with_scores
            existing_chunks = {chunk for chunk, _ in relevant_chunks_with_scores}
            remaining_chunks = [(chunk, score) for chunk, score in all_scored_chunks if chunk not in existing_chunks]

            # Add random chunks until we have enough or run out
            import random
            random.shuffle(remaining_chunks)
            relevant_chunks_with_scores.extend(remaining_chunks[:num_chunks - len(relevant_chunks_with_scores)])

        print(f"Found {len(relevant_chunks_with_scores)} relevant chunks")

    except Exception as e:
        print(f"Error retrieving relevant chunks: {e}")
        import traceback
        traceback.print_exc()
        # Return empty list if search fails
        relevant_chunks_with_scores = []
        print("Returning empty list of relevant chunks due to search errors")
    finally:
        # Clean up resources
        gc.collect()

    # Cache the results if we got any
    if relevant_chunks_with_scores:
        try:
            print(f"Caching {len(relevant_chunks_with_scores)} relevant chunks")
            cache.set(cache_key, relevant_chunks_with_scores, timeout=3600)  # Cache for 1 hour
            print(f"Successfully cached relevant chunks")
        except Exception as cache_error:
            print(f"Error caching relevant chunks: {cache_error}")
            # Continue even if caching fails

    return relevant_chunks_with_scores



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

def generate_quiz_from_pdf(pdf_obj, num_questions, difficulty, task_id=None):
    """
    Generate a quiz from a PDF file with progress tracking.
    Optimized for memory efficiency with streaming processing and explicit cleanup.
    Enhanced to handle very large PDFs (100MB+) efficiently.

    Args:
        pdf_obj: The UploadedPDF object
        num_questions: Number of questions to generate
        difficulty: Difficulty level of the questions
        task_id: Optional Celery task ID for progress tracking

    Returns:
        dict: A dictionary containing the quiz object, content, and token usage
    """
    # Helper function to update progress if task_id is provided
    def update_sub_progress(step, message, sub_step=None, total_sub_steps=None):
        try:
            if task_id:
                from .tasks import update_progress
                print(f"Updating progress: step={step}, message={message}, sub_step={sub_step}, total_sub_steps={total_sub_steps}")

                if sub_step is not None and total_sub_steps is not None:
                    # Calculate overall progress within the current main step
                    # Main steps are 1-5, and we're in step 3 or 4 with sub-steps
                    if step == 3:  # Processing text step
                        main_step = 3
                        progress = 2 + (sub_step / total_sub_steps)
                    elif step == 4:  # Generating questions step
                        main_step = 4
                        progress = 3 + (sub_step / total_sub_steps)
                    else:
                        main_step = step
                        progress = step

                    print(f"Calculated progress: {progress}")
                    update_progress(task_id, progress, 5, f"{message} ({sub_step}/{total_sub_steps})")
                else:
                    print(f"Using step as progress: {step}")
                    update_progress(task_id, step, 5, message)
        except Exception as e:
            print(f"Error updating progress: {e}")
            # Continue execution even if progress update fails

    try:
        # Extract text with caching using the optimized method
        update_sub_progress(2, "Extracting text from PDF...")
        full_text = extract_text_from_pdf(pdf_obj.file, pdf_obj)

        # For very large PDFs, we'll use a different approach to chunking
        is_large_pdf = len(full_text) > 1000000  # Consider PDFs with > 1MB of text as "large"
        print(f"PDF text size: {len(full_text)} characters, treating as {'large' if is_large_pdf else 'normal'} PDF")

        if is_large_pdf:
            update_sub_progress(3, "Processing large PDF with optimized chunking...", 1, 4)

            # For large PDFs, we'll use a larger chunk size with more overlap
            # to capture more context while reducing the total number of chunks
            chunk_size = 2000
            overlap = 400

            # Use the generator directly to avoid loading all chunks into memory
            chunks_generator = chunk_text(full_text, max_chunk_size=chunk_size, overlap=overlap)

            # Store chunks in files for later retrieval, using a streaming approach
            update_sub_progress(3, "Storing text chunks for large PDF...", 2, 4)
            chunks_dir = store_text_chunks(chunks_generator, pdf_obj.id, task_id=task_id)

            # Free up memory after text chunk processing
            full_text = None  # Clear the full text immediately for large PDFs
            gc.collect()

            # For large PDFs, we'll retrieve more chunks to ensure good coverage
            update_sub_progress(3, "Finding relevant content in large PDF...", 4, 4)
            query = f"Generate {num_questions} {difficulty} multiple-choice questions about the main topics and key concepts"

            # Retrieve more chunks for large PDFs to ensure good coverage
            num_chunks = min(10, num_questions * 2)  # Scale with number of questions, but cap at 10
            relevant_chunks_with_scores = retrieve_relevant_chunks(query, pdf_obj.id, num_chunks=num_chunks)
            relevant_chunks = [chunk for chunk, score in relevant_chunks_with_scores]
        else:
            # For normal-sized PDFs, use the original approach
            update_sub_progress(3, "Chunking text...", 1, 4)
            chunks_generator = chunk_text(full_text, max_chunk_size=500, overlap=200)

            # Store chunks in files for later retrieval
            update_sub_progress(3, "Processing text chunks...", 2, 4)
            chunks_dir = store_text_chunks(chunks_generator, pdf_obj.id, task_id=task_id)

            # Free up memory after text chunk processing
            gc.collect()

            # Find relevant content
            update_sub_progress(3, "Finding relevant content...", 4, 4)
            query = f"Generate {num_questions} {difficulty} multiple-choice questions about the main topics"
            relevant_chunks_with_scores = retrieve_relevant_chunks(query, pdf_obj.id, num_chunks=5)
            relevant_chunks = [chunk for chunk, score in relevant_chunks_with_scores]

        # Combine the relevant chunks to create the content for the quiz
        update_sub_progress(4, "Preparing content for quiz generation...", 1, 3)
        if relevant_chunks_with_scores:
            # For large PDFs, we need to be more selective about which content to include
            if is_large_pdf and len(relevant_chunks_with_scores) > 5:
                # Sort chunks by importance score (descending)
                sorted_chunks = sorted(relevant_chunks_with_scores, key=lambda x: x[1], reverse=True)

                # Take the top 5 most important chunks (or fewer if not enough)
                selected_chunks = [chunk for chunk, score in sorted_chunks[:5]]

                optimized_content = "\n\n".join(selected_chunks)
            else:
                # For normal PDFs or if few relevant chunks, just join them
                optimized_content = "\n\n".join(relevant_chunks)

            # Trim if necessary
            if len(optimized_content) > 3000:
                # For large PDFs, be smarter about trimming to keep the most relevant content
                if is_large_pdf:
                    # Split into sentences and prioritize sentences with important keywords
                    sentences = re.split(r'(?<=[.!?])\s+', optimized_content)
                    important_keywords = [
                        'important', 'significant', 'key', 'main', 'critical', 'essential',
                        'fundamental', 'crucial', 'vital', 'primary', 'major', 'central',
                        'define', 'definition', 'concept', 'theory', 'principle', 'method'
                    ]

                    # Score sentences by importance
                    scored_sentences = []
                    for i, sentence in enumerate(sentences):
                        score = 0
                        for keyword in important_keywords:
                            if keyword in sentence.lower():
                                score += 1
                        scored_sentences.append((i, score, sentence))

                    # Sort by score (descending) then by original order (ascending) to maintain context
                    scored_sentences.sort(key=lambda x: (-x[1], x[0]))

                    # Take sentences until we reach ~3000 characters
                    selected_sentences = []
                    char_count = 0
                    for _, _, sentence in scored_sentences:
                        if char_count + len(sentence) + 1 <= 3000:  # +1 for space
                            selected_sentences.append(sentence)
                            char_count += len(sentence) + 1
                        else:
                            break

                    # Sort back to original order to maintain context
                    selected_sentences.sort(key=lambda s: sentences.index(s))
                    optimized_content = " ".join(selected_sentences)
                else:
                    # Simple trimming for normal PDFs
                    optimized_content = optimized_content[:3000]
        else:
            # Fall back to the original approach if keyword matching doesn't return relevant chunks
            if full_text and len(full_text) > 3000:
                optimized_content = full_text[:3000]
            else:
                optimized_content = full_text or ""

        # Free up memory after content preparation
        relevant_chunks_with_scores = None
        if full_text and len(full_text) > 10000:  # Only clear for large documents
            full_text = None
        gc.collect()

        # Create question templates locally
        update_sub_progress(4, "Creating question templates...", 2, 3)
        templates = create_question_templates(num_questions, difficulty)
        template_examples = "\n".join(templates[:3])  # Send just a few examples to save tokens

        # Initialize the LLM with a lower temperature for more focused responses
        update_sub_progress(4, "Generating quiz questions with AI...", 3, 3)
        llm = ChatOpenAI(
            temperature=0.5,
            # Add streaming for memory efficiency
            streaming=True
        )

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

        # Free up memory before API call
        optimized_content = None
        templates = None
        template_examples = None
        gc.collect()

        # Track token usage
        with get_openai_callback() as cb:
            response = llm(final_prompt)

            # Free up memory after API call
            final_prompt = None
            gc.collect()

            # Save token usage to database
            token_usage = TokenUsage.objects.create(
                prompt_tokens=cb.prompt_tokens,
                completion_tokens=cb.completion_tokens,
                total_tokens=cb.total_tokens
            )

            # Parse the quiz content into structured data
            update_sub_progress(5, "Parsing quiz content...")
            parsed_questions = parse_quiz_content(response.content)

            # Create the quiz
            quiz = Quiz.objects.create(
                pdf=pdf_obj,
                num_questions=num_questions,
                difficulty=difficulty
            )

            # Create the questions in batches to reduce memory usage
            batch_size = 5
            for i in range(0, len(parsed_questions), batch_size):
                batch = parsed_questions[i:i+batch_size]

                # Update progress every batch
                update_sub_progress(5, f"Saving questions {i+1}-{min(i+batch_size, len(parsed_questions))}/{len(parsed_questions)}...")

                for q_data in batch:
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

                # Free memory after each batch
                gc.collect()

            # Cache the quiz result in Redis
            cache_key = f"quiz_result_{quiz.id}"
            cache_data = json.dumps([quiz.id, response.content, token_usage.id])
            cache.set(cache_key, cache_data, timeout=86400)  # Cache for 24 hours

            # Final cleanup
            parsed_questions = None
            response = None
            gc.collect()

            # Return both the quiz object, raw content, and token usage
            return {
                'quiz': quiz,
                'content': cache.get(cache_key),
                'token_usage': token_usage
            }
    except Exception as e:
        # Ensure memory is freed even if an error occurs
        gc.collect()
        raise
