from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from rest_framework import generics,status
from .models import Cours
from .serializer import CoursSerializer
from rest_framework.response import Response
from rest_framework.views import APIView

#get cour
class CoursList(generics.ListCreateAPIView):
    queryset = Cours.objects.all()
    serializer_class = CoursSerializer


#cour by id get, update, delete
class CoursDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Cours.objects.all()
    serializer_class = CoursSerializer

def helloWorld(HttpRequest):
    return HttpResponse("Hello World")

