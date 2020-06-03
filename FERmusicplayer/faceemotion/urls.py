from django.urls import path
from django.views.generic import TemplateView
from . import views

app_name = "fermodel"

urlpatterns = [
    path('', TemplateView.as_view(template_name='faceemotion/emotion_detect.html'), name='camera'),
    path('recognize/', views.picture_ajax_upload, name='recognize'),
]
