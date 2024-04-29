from typing import Any
from django.shortcuts import render, get_object_or_404
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from django.db.models import (
    Q,
    F,
    ExpressionWrapper
)  # Q objects to create an 'OR' statament, F objects to create col1==col2
from django_filters.rest_framework import DjangoFilterBackend
from home.pagination import DefaultPagination # for generic filtering
from home.models import Prediction
from rest_framework.mixins import ListModelMixin, CreateModelMixin
from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.viewsets import ModelViewSet
from rest_framework.filters import SearchFilter, OrderingFilter
#  from rest_framework.pagination import PageNumberPagination
from .serializers import PredictionSerializer

# A view function is a function that takes a request and returns a response.
# It's a request handler and in some framewroks is called an action.

# http://127.0.0.1:8000/home/hello/

class PredictionViewSet(ModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter] # SearchFilter: to search string columns
    filterset_fields = ['id_match', 'date'] # http://127.0.0.1:8000/home/predictions/?id_match=MNMbQMK1
    # http://127.0.0.1:8000/home/predictions/?date=2024-02-24%2014:30:00.000000
    pagination_class = DefaultPagination
    search_fields = ['predicted_result'] # http://127.0.0.1:8000/home/predictions/?search=1
    ordering_filter = ['prob_class_1']

    def get_serializer_context(self):
        return {'request': self.request}

def index(request):
    return render(request, 'index.html')

def predicciones(request):
    prob_home = ExpressionWrapper(F('prob_home_bm') * 100, output_field=models.IntegerField())
    queryset = Prediction.objects.annotate(prob_home=prob_home)
    return render(request, 'predicciones.html', {"name": "Caro & Nacho Co", "predictions": queryset})

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

