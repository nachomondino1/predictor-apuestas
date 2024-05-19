from django.urls import include, path
from . import views
# from rest_framework.routers import DefaultRouter # SimpleRouter
from rest_framework_nested import routers
from pprint import pprint

router = routers.DefaultRouter()

# Routers: We registry the view set here
router.register('predictions', views.PredictionViewSet, basename='predictions')
router.register('images', views.ImageViewSet, basename='media/images')
pprint(router.urls)

# prediction_router = routers.NestedDefaultRouter(router, 'predictions', lookup='pred')

# URLConf
# urlpatterns = [
#     path('', include(router.urls)),
#     path('hello/', views.say_hello),
#     path('predicciones/', views.view_predicciones),
# ]
urlpatterns = [
    path('', views.index, name='index'),
    path('predicciones/', views.predicciones, name='predicciones'), # path('predicciones/', views.predicciones, name='predicciones'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('legal/', views.legal, name='legal'),
    # path('hello/', views.say_hello),
    # path('predictions/', views.PredictionList.as_view()),
    # path('predictions/<str:pk>/', views.PredictionDetail.as_view()) # <str:id> <str:team_name> <str:id_match>
]

"""
urlpatterns = [
    path('', include(router.urls)),
    path('hello/', views.say_hello),
    path('predicciones/', views.PredictionViewSet.as_view({'get': 'list'})),
>>>>>>> staging
]
"""