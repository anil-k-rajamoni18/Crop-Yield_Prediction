from django import forms
from django.forms.formsets import formset_factory
'''
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)

'''
class InputForm(forms.Form):
   # Text = forms.CharField(max_length=80, widget=forms.TextInput(attrs={'class': 'autocomplete'}))
    temp=forms.FloatField(max_value=2000,min_value=0,
    	widget=forms.NumberInput(attrs={'class': 'form-control'}),
    	help_text='temperature in celcius (12-50)')
    rainfall=forms.FloatField(max_value=2000,min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='average rainfall (mm) (400-1100)')
    soil=forms.FloatField(max_value=14,min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='PH value of Soil (3-10)')
    crop=forms.CharField(max_length=100,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Name of the Crop')
    land=forms.CharField(max_length=100,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Number of Acres')




    