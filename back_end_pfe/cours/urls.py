from django import urls
from django.urls import path
from .views import CoursList, CoursDetail, helloWorld

urlpatterns = [
    path('cours/', CoursList.as_view(), name='cours-list'),
    path('cours/<int:pk>', CoursDetail.as_view(), name='cours-detail')
    # path('hello/', helloWorld, name='hello-world'
]