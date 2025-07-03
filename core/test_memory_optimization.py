import os
import sys
import gc
import psutil
import time
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from .models import UploadedPDF
from .quiz_generator import extract_text_from_pdf, chunk_text, get_text_chunks, store_text_embeddings, retrieve_relevant_chunks, generate_quiz_from_pdf

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory_usage(label):
    """Log current memory usage with a label"""
    memory_mb = get_memory_usage()
    print(f"{label}: {memory_mb:.2f} MB")
    return memory_mb

def test_memory_optimization():
    """Test memory optimization in quiz generation process"""
    print("\n=== Testing Memory Optimization ===")
    
    # Create a test PDF file
    test_pdf_path = os.path.join(os.path.dirname(__file__), 'test_data', 'sample.pdf')
    if not os.path.exists(test_pdf_path):
        print(f"Test PDF not found at {test_pdf_path}")
        print("Please create a test PDF file for testing")
        return
    
    # Start with a clean slate
    gc.collect()
    initial_memory = log_memory_usage("Initial memory usage")
    
    # Test PDF extraction
    print("\n--- Testing PDF Text Extraction ---")
    with open(test_pdf_path, 'rb') as f:
        pdf_content = f.read()
    
    pdf_file = SimpleUploadedFile("test.pdf", pdf_content, content_type="application/pdf")
    pdf_obj = UploadedPDF.objects.create(file=pdf_file)
    
    log_memory_usage("Memory after PDF upload")
    
    # Extract text
    start_time = time.time()
    text = extract_text_from_pdf(pdf_obj.file, pdf_obj)
    extraction_time = time.time() - start_time
    
    memory_after_extraction = log_memory_usage("Memory after text extraction")
    print(f"Extraction time: {extraction_time:.2f} seconds")
    print(f"Extracted text length: {len(text)} characters")
    
    # Test chunking
    print("\n--- Testing Text Chunking ---")
    start_time = time.time()
    
    # Test generator-based chunking
    chunks_generator = chunk_text(text, max_chunk_size=500, overlap=100)
    chunking_time = time.time() - start_time
    
    memory_after_generator = log_memory_usage("Memory after creating chunks generator")
    
    # Count chunks without materializing the full list
    chunk_count = 0
    for _ in chunks_generator:
        chunk_count += 1
    
    print(f"Chunking time: {chunking_time:.2f} seconds")
    print(f"Number of chunks: {chunk_count}")
    
    # Force garbage collection
    gc.collect()
    log_memory_usage("Memory after garbage collection")
    
    print("\n=== Memory Optimization Test Complete ===")
    print(f"Memory increase during test: {memory_after_extraction - initial_memory:.2f} MB")
    
    # Clean up
    pdf_obj.delete()
    
if __name__ == "__main__":
    # This will only run when script is executed directly, not when imported
    test_memory_optimization()