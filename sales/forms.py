from django import forms
from .models import RegistrationDetails,SalesDetails

class UserDetails(forms.ModelForm):
    class Meta:
        model = RegistrationDetails
        fields = '__all__'


class SalesDetailsForm(forms.ModelForm):
    class Meta:
        model = SalesDetails
        fields = '__all__'