# core/forms.py
from django import forms

class QuizForm(forms.Form):
    pdf_file = forms.FileField(
        required=False,
        widget=forms.FileInput(attrs={'class': 'form-control file-input'}),
        help_text="Upload a PDF document to generate questions from its content"
    )
    num_questions = forms.IntegerField(
        min_value=1, 
        max_value=50,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    difficulty = forms.ChoiceField(
        choices=[
            ('easy', 'Easy'), 
            ('medium', 'Medium'), 
            ('hard', 'Hard')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    exam_style = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        help_text="Enter a famous exam, test, or book style (e.g., 'SAT', 'GMAT', 'AP Biology', 'Harry Potter')"
    )
    include_long_answer = forms.BooleanField(
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text="Include questions where you need to type a long answer"
    )

    def clean(self):
        cleaned_data = super().clean()
        pdf_file = cleaned_data.get('pdf_file')
        exam_style = cleaned_data.get('exam_style')

        if pdf_file and exam_style:
            raise forms.ValidationError(
                "Please provide either a PDF file OR an exam style, not both."
            )

        if not pdf_file and not exam_style:
            raise forms.ValidationError(
                "Please provide either a PDF file OR an exam style."
            )

        return cleaned_data
