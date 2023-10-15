from django.urls import path
from . import views

urlpatterns = [
    path('', views.welcome, name="welcome"),
    path('order', views.fun2, name="fun2"),
    path('ticket', views.fun3, name="fun3")
]
