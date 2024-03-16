from django.urls import path
from . import views

# URLConf
urlpatterns = [
    path('hello/', views.say_hello),
    path('predictions/', views.prediction_list),
    path('predictions/<str:id_match>/', views.prediction_detail) # <str:id> <str:team_name>
]