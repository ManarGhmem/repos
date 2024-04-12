from django import urls
from django.urls import path
from .views import project_totalList, project_totalDetail

urlpatterns=[
    path('project_total/',project_totalList.as_view(),name='project_total-list'),
    path('project_total/<int:pk>',project_totalDetail.as_view(),name='project_total-detail')
]