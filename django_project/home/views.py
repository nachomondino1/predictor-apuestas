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


def say_hello(request):

    # pk='zRQ2Ptz8'
    # queryset api: field lookups -> you can search the lookups types.
    # queryset = Predictions.objects.filter(odds_home__range=(0,2)) # pk:primeary key. We don't have to remember the name of the pk field.
    # queryset = Predictions.objects.filter(id_team_home__contains='Man')
    # queryset = Predictions.objects.filter(date__day=10)
    # queryset = Predictions.objects.filter(Q(odds_home__lt=10) | Q(odds_home__gt=5))
    """ 
    queryset = Predictions.objects.values_list(
        "id_team_home", "id_team_away", "predicted_result"
    ).distinct()  # only read some columns from the table

    queryset = Predictions.objects.select_related() --> join, the other side has 1 values
    queryset = Predictions.objects.prefetch_related() --> join, the other side has n values

    ExpressionWrapper --> More complex queries
    objets = returns a manager = it's a gateway to the db
    """
    prob_home = ExpressionWrapper(F('prob_home_bm') * 100, output_field=models.IntegerField())
    queryset = Prediction.objects.annotate(prob_home=prob_home)
    # queryset = Predictions.objects.all()
    return render(
        request, "hello.html", {"name": "Caro & Nacho Co", "predictions": queryset}
    )


""" 
@api_view() # the request will be an instance of the framework
def prediction_list(request):
    queryset = Prediction.objects.all()
    serializer = PredictionSerializer(queryset, many=True)
    return Response(serializer.data)  # return HttpResponse('ok')
    
@api_view()
def prediction_detail(request, team_name): # id = MLiWiTUt
    prediction = Prediction.objects.get(id_team_home__contains=team_name)
    serializer = PredictionSerializer(prediction)
    return Response(serializer.data)
"""
"""
OPTION 1: You can write the try and except every time or use get_object_or_404
def prediction_detail(request, id_match): # id = MLiWiTUt
    try: 
        prediction = Prediction.objects.get(id_match=id_match)
        serializer = PredictionSerializer(prediction)
        return Response(serializer.data)
    except Prediction.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

# OPTION 2: use get_object_or_404 that has the try and except inside
# Function based views
@api_view(['GET', 'POST']) # http://127.0.0.1:8000/home/predictions/MLiWiTUt/
def prediction_detail(request, id_match): # id = MLiWiTUt
    if request.method == 'GET':
        prediction = get_object_or_404(Prediction, id_match=id_match)
        serializer = PredictionSerializer(prediction)
        return Response(serializer.data)
    elif request.method == 'POST':
        serializer = PredictionSerializer(data=request.data) # deserializer
        serializer.is_valid(raise_exception=True)
        serializer.save()
        # serializer.validated_data
        return Response('Ok')
"""

# OPTION 3: use get_object_or_404 that has the try and except inside
"""
Option 1: APIView + functions
class PredictionList(APIView): 
    def get(self, request):
        queryset = Prediction.objects.all()
        serializer = PredictionSerializer(queryset, many=True)
        return Response(serializer.data)  # return HttpResponse('ok')
    
    def post(self,request):
        serializer = PredictionSerializer(data=request.data) # deserializer
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response('Ok')
"""

""" 
class PredictionDetail(APIView):
    def get(self, request, id_match):
        prediction = get_object_or_404(Prediction, id_match=id_match)
        serializer = PredictionSerializer(prediction)
        return Response(serializer.data)
 """
"""
Option 3: ListCreateAPIView

class PredictionList(ListCreateAPIView):  # with generic views

    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
  
    def get_serializer_context(self):
        return {'request': self.request}

class PredictionDetail(RetrieveUpdateDestroyAPIView):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer

We can change to:If you need some logic!
    def get_queryset(self):
        return Prediction.objects.all()
    def get_serializer_class(self):
        return PredictionSerializer
"""

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
    
def view_predicciones(request):
    
    prob_home = ExpressionWrapper(F('prob_home_bm') * 100, output_field=models.IntegerField())
    queryset = Prediction.objects.annotate(prob_home=prob_home)
    return render(
        request, "predicciones.html", {"name": "Caro & Nacho Co", "predictions": queryset}
    )