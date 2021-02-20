from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
    path('',views.locationView,name="location"),
    path('home',views.home,name="home")
]