from django.urls import path
from django.views.generic import TemplateView
from . import views

app_name = "musicplayer"

urlpatterns = [
    path('musics-list/', views.MusicsList.as_view(), name='music_list'),
    path('player/', views.PlayerView.as_view(), name='music_player'),
    path('', views.index_redirect),
]
