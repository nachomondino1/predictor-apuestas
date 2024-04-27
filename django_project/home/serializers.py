from rest_framework import serializers
from home.models import Prediction

# class PredictionSerializer(serializers.Serializer):
class PredictionSerializer(serializers.ModelSerializer):
    """ This values are the one we are going to send when we have an API request!

    This fields are the ones the poeple can see

    Args:
        serializers (_type_): _description_
    """
    winner_team_name = serializers.SerializerMethodField()

    def get_winner_team_name(self,Prediction): # calculated field
        if Prediction.prob_home_bm > Prediction.prob_away_bm:
            return Prediction.id_team_home
        elif Prediction.prob_home_bm < Prediction.prob_away_bm:
            return Prediction.id_team_away
        else:
            return 'draw'

    class Meta: # http://127.0.0.1:8000/home/predictions/?date=2024-02-24&id_match=&ordering=time
        model = Prediction
        fields = ['id_match', 'date', 'time', 'id_team_home', 'id_team_away','prob_home_bm','prob_draw_bm','prob_away_bm', 'predicted_result','winner_team', 'winner_team_name']
    """
    Other option: 
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