from typing import Any
from django.contrib import admin
from django.db.models.query import QuerySet
from . import models

# Register your models here.

@admin.register(models.Prediction)
class PredictionAdmin(admin.ModelAdmin): # how we want to view or edit our products
    # autocomplete_fields = ['id_team_home','id_team_away']
    list_display = ['date', 'id_team_home', 'id_team_away', 'predicted_result', 'prediction_winner']
    # list_editable = ['predicted_result']
    # list_per_page = 10
    list_filter = ['date','predicted_result']
    ordering = ['id_team_home', 'id_team_away']
    search_fields = ['id_team_home__istartswith', 'id_team_away__istartswith']
    
    # @admin.display(ordering='prediction')
    def prediction_winner(self, prediction):
        if prediction.predicted_result == '0' or  prediction.predicted_result == 'X':
            return 'Draw'
        elif prediction.predicted_result == '1':
            return 'Home'
        else:
            return 'Away'