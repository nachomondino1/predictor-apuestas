from django.urls import include, path
from . import views
from rest_framework.routers import DefaultRouter # SimpleRouter
from pprint import pprint

# Routers: We registry the view set here
router = DefaultRouter()
router.register('predictions', views.PredictionViewSet)
pprint(router.urls)
# http://127.0.0.1:8000/home/predictions.json --> in json format


# URLConf
"""
urlpatterns = [
    # path('hello/', views.say_hello),
    # path('predictions/', views.PredictionList.as_view()),
    # path('predictions/<str:pk>/', views.PredictionDetail.as_view()) # <str:id> <str:team_name> <str:id_match>
]
 """
urlpatterns = [
    path('', include(router.urls)),
    path('hello/', views.say_hello),
]

