from django.urls import include, path
from . import views
# from rest_framework.routers import DefaultRouter # SimpleRouter
from rest_framework_nested import routers
from pprint import pprint

router = routers.DefaultRouter()

# Routers: We registry the view set here
router = routers.DefaultRouter()
router.register('predictions', views.PredictionViewSet, basename='predictions')
pprint(router.urls)
# http://127.0.0.1:8000/home/predictions.json --> in json format

# prediction_router = routers.NestedDefaultRouter(router, 'predictions', lookup='pred')

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
    path('predicciones/', views.view_predicciones),
]