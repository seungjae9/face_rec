from django.urls import path
from . import views

app_name = 'page'

urlpatterns = [
    path('', views.index, name='index'),
    path('who/', views.who, name='who'),
    path('merge/', views.merge, name='merge'),
    path('mixed/<str:name>', views.mixed, name='mixed'),
    path('mixed_all/<str:name1>/<str:name2>/<str:name3>', views.mixed_all, name='mixed_all'),
]