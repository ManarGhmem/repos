from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from rest_framework import generics,status
from .models import project_total
from .serializer import project_totalSerializer
from rest_framework.response import Response
from rest_framework.views import APIView

# Create your views here.
class project_totalList(generics.ListCreateAPIView):
    queryset= project_total.objects.all()
    serializer_class=project_totalSerializer


class project_totalDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset=project_total.objects.all()
    serializer_class=project_totalSerializer