# Memory Optimization for PDF Quiz Generator

This document outlines the memory optimization techniques implemented in the PDF Quiz Generator application to improve performance and reduce memory usage, especially when processing large PDF files.

## Overview of Optimizations

The following memory optimization techniques have been implemented:

1. **Streaming PDF Processing**: Changed from loading entire PDFs into memory to processing one page at a time
2. **Generator-based Text Chunking**: Replaced list-based chunking with generator-based approach
3. **Batch Processing for Embeddings**: Process embeddings in small batches with memory cleanup between batches
4. **Explicit Resource Cleanup**: Added explicit variable clearing and garbage collection at key points
5. **Streaming LLM Responses**: Enabled streaming mode for OpenAI API calls
6. **Proper Exception Handling**: Ensured resources are cleaned up even when errors occur
7. **Conditional Memory Management**: Clear large variables only when necessary based on size

## Detailed Optimizations

### PDF Text Extraction

The `extract_text_from_pdf` function was optimized to:
- Process one page at a time instead of loading the entire PDF
- Store page text in a list and only join when needed
- Run garbage collection periodically during processing
- Clear temporary variables after use

```python
# Process one page at a time to reduce memory usage
all_text = []
for i, page in enumerate(reader.pages):
    page_text = page.extract_text()
    all_text.append(page_text)
    
    # Free up memory after processing each page
    if i % 10 == 0:  # Every 10 pages
        gc.collect()
```

### Text Chunking

The `chunk_text` function was converted to a generator that yields chunks one at a time:
- Yields chunks instead of building a list
- Runs garbage collection periodically
- Added a wrapper function for backward compatibility

```python
def chunk_text(text, max_chunk_size=500, overlap=100):
    """
    Split text into chunks of maximum size with overlap between chunks.
    Uses a generator approach to reduce memory usage.
    """
    start = 0
    text_length = len(text)

    while start < text_length:
        # ... chunk calculation logic ...
        yield text[start:end]
        # ... update start position ...
        
        # Run garbage collection periodically
        if start % (max_chunk_size * 10) < max_chunk_size:
            gc.collect()
```

### Embedding Generation

The `store_text_embeddings` function was optimized to:
- Support both list and generator inputs
- Process chunks in smaller batches (reduced from 5 to 3)
- Clear batch variables after processing
- Run garbage collection after each batch
- Add proper progress tracking for both list and generator inputs

```python
# Process chunks in a streaming fashion
batch_texts = []
batch_metadatas = []
# ... process chunks one by one ...
for chunk in text_chunks:
    batch_texts.append(chunk)
    # ... when batch is full, process it ...
    if len(batch_texts) >= batch_size:
        # ... process batch ...
        # Clear the batch and run garbage collection
        batch_texts = []
        batch_metadatas = []
        gc.collect()
```

### Vector Search

The `retrieve_relevant_chunks` function was optimized to:
- Initialize variables to None for proper cleanup
- Add a finally block to ensure resources are cleaned up
- Clear results after extracting relevant chunks
- Run garbage collection at appropriate points

```python
try:
    # ... search logic ...
    # Clear results to free memory
    results = None
    gc.collect()
except Exception as e:
    # ... error handling ...
finally:
    # Clean up resources
    embeddings = None
    chroma_client = None
    chroma_db_client = None
    gc.collect()
```

### Quiz Generation

The `generate_quiz_from_pdf` function was optimized to:
- Use generator-based chunking
- Add explicit memory cleanup points
- Use streaming mode for LLM
- Clear variables when no longer needed
- Process questions in batches
- Add try-except for proper cleanup on errors

```python
try:
    # ... quiz generation logic with cleanup points ...
    # Free up memory after content preparation
    relevant_chunks = None
    if len(full_text) > 10000:  # Only clear for large documents
        full_text = None
    gc.collect()
    
    # ... more processing with cleanup points ...
except Exception as e:
    # Ensure memory is freed even if an error occurs
    gc.collect()
    raise
```

## Testing

A test script (`test_memory_optimization.py`) was created to verify the memory optimizations:
- Measures memory usage at different stages
- Tests PDF text extraction
- Tests generator-based chunking
- Logs memory usage, timing, and other metrics

## Conclusion

These memory optimizations significantly reduce the memory footprint of the application, especially when processing large PDFs. The application can now handle larger documents more efficiently without running out of memory.

The optimizations maintain the same functionality while improving performance and resource usage. Future development should continue to follow these memory-efficient patterns, especially when dealing with large files or resource-intensive operations.