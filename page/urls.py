from django.urls import path
from . import views

app_name = 'page'

urlpatterns = [
    path('', views.index, name='index'),
    path('who/', views.who, name='who'),
    path('merge/', views.merge, name='merge'),
    path('mixed/<str:name>', views.mixed, name='mixed'),
]