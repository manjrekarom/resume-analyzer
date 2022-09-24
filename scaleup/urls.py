from django.urls import path
from . import views

urlpatterns = [
    path("", views.input, name="input"),
    path("analyse", views.analyse, name="analyse")
]