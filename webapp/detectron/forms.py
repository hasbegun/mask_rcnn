from django import forms
from .models import UploadFile


class UploadFileForm(forms.Form):
    upload_file = forms.FileField()


class UploadFileModelForm(forms.ModelForm):
    class Meta:
        model = UploadFile
        fields = ('desctiption', 'upload_file')
