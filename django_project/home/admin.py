from django.contrib import admin
from . import models

# Register your models here.

"""
    @admin.register(models.Predictions) # decorator or 
class PredictionsAdmin(admin.ModelAdmin): # how we want to view or edit our products
    list_display = ['date', 'id_team_home', 'id_team_away', 'predicted_result', 'prediction_winner']
    # list_editable = ['predicted_result']
    # list_per_page = 10

    # @admin.display(ordering='prediction')
    def prediction_winner(self, prediction):
        if prediction.prob_home_bm > prediction.prob_away_bm:
            return 'Home'
        return 'Away'

"""
@admin.register(models.Prediction)
class PredictionAdmin(admin.ModelAdmin): # how we want to view or edit our products
    list_display = ['date', 'id_team_home', 'id_team_away', 'predicted_result']
    # list_editable = ['predicted_result']
    # list_per_page = 10