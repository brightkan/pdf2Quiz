# core/forms.py
from django import forms

class QuizForm(forms.Form):
    pdf_file = forms.FileField(
        widget=forms.FileInput(attrs={'class': 'form-control'})
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
