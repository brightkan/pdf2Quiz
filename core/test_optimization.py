import os
import sys
import django
import io
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.conf import settings

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

from core.quiz_generator import generate_quiz_from_pdf
from core.models import TokenUsage, UploadedPDF

def create_test_pdf():
    """Create a simple PDF file for testing."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Test PDF for Quiz Generation")
    c.drawString(100, 700, "This is a sample PDF with some educational content.")
    c.drawString(100, 650, "Machine Learning is a subset of artificial intelligence.")
    c.drawString(100, 600, "It focuses on the development of algorithms that can learn from data.")
    c.drawString(100, 550, "Supervised learning is when the model is trained on labeled data.")
    c.drawString(100, 500, "Unsupervised learning is when the model finds patterns in unlabeled data.")
    c.drawString(100, 450, "Reinforcement learning involves an agent learning to make decisions.")
    c.drawString(100, 400, "Deep learning uses neural networks with many layers.")
    c.drawString(100, 350, "Natural Language Processing (NLP) is used for text analysis.")
    c.drawString(100, 300, "Computer Vision is used for image and video analysis.")
    c.save()

    buffer.seek(0)
    return buffer

def test_optimization():
    """Test the optimization of the quiz generation process."""
    # Create a test PDF
    pdf_buffer = create_test_pdf()

    # Create an InMemoryUploadedFile from the buffer
    pdf_file = InMemoryUploadedFile(
        pdf_buffer,
        'file',
        'test.pdf',
        'application/pdf',
        len(pdf_buffer.getvalue()),
        None
    )

    # Generate a quiz with the test PDF
    print("Generating first quiz...")
    result1 = generate_quiz_from_pdf(pdf_file, 3, 'easy')
    token_usage1 = result1['token_usage']

    # Reset the file pointer
    pdf_file.seek(0)

    # Generate another quiz with the same PDF to test caching
    print("Generating second quiz (should use cached text)...")
    result2 = generate_quiz_from_pdf(pdf_file, 3, 'easy')
    token_usage2 = result2['token_usage']

    # Print token usage for both runs
    print(f"First run token usage: {token_usage1.total_tokens}")
    print(f"Second run token usage: {token_usage2.total_tokens}")

    # Check if the PDF text was cached
    pdf_obj = UploadedPDF.objects.get(id=result2['quiz'].pdf.id)
    print(f"PDF text cached: {bool(pdf_obj.extracted_text)}")

    # Print the questions generated
    print("\nGenerated Questions:")
    for i, question in enumerate(result2['quiz'].questions.all()):
        print(f"\nQuestion {i+1}: {question.text}")
        print(f"A: {question.option_a}")
        print(f"B: {question.option_b}")
        print(f"C: {question.option_c}")
        print(f"D: {question.option_d}")
        print(f"Correct: {question.correct_option}")
        print(f"Topic: {question.topic}")

if __name__ == "__main__":
    test_optimization()
