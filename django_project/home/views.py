from django.shortcuts import render
from django.http import HttpResponse
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from django.db.models import (
    Q,
    F,
    ExpressionWrapper
)  # Q objects to create an 'OR' statament, F objects to create col1==col2
from home.models import Prediction

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
