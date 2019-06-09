from django.urls import path
from . import views

app_name = 'detectron'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.file_upload, name='file_upload'),
    path('form/', views.model_form_upload, name='model_form_upload'),
]
