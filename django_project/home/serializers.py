from rest_framework import serializers
from home.models import Prediction

# class PredictionSerializer(serializers.Serializer):
class PredictionSerializer(serializers.ModelSerializer):
    """ This values are the one we are going to send when we have an API request!

    Args:
        serializers (_type_): _description_
    """
    class Meta:
        model = Prediction
        fields = ['id_match', 'date', 'id_team_home', 'id_team_away','prob_home_bm','prob_draw_bm','prob_away_bm', 'predicted_result','winner_team']
    """
    id_match = serializers.CharField()
    date = serializers.DateTimeField()
    id_team_home = serializers.CharField(max_length=100) # models.ForeignKey(TeamsFlashscore, related_name='id_team_home', on_delete=models.PROTECT)
    id_team_away = serializers.CharField(max_length=100) # models.ForeignKey(TeamsFlashscore, related_name='id_team_away', on_delete=models.PROTECT)
    prob_home_bm =  serializers.DecimalField(max_digits=4, decimal_places=2)
    prob_draw_bm =  serializers.DecimalField(max_digits=4, decimal_places=2)
    prob_away_bm =  serializers.DecimalField(max_digits=4, decimal_places=2)
    predicted_result = serializers.CharField(max_length=5)  
    
    """
    winner_team = serializers.SerializerMethodField(method_name='get_winner_team') # custom 

    def get_winner_team(self, prediction: Prediction):
        if prediction.predicted_result == "x" or prediction.predicted_result == "0":
            return "draw"
        elif prediction.predicted_result == "1":
            return "home"
        elif prediction.predicted_result == "2":
            return "away"