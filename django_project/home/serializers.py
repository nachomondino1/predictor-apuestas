from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    """ This values are the one we are going to send when we have an API request!

    Args:
        serializers (_type_): _description_
    """
    id_match = serializers.CharField()
    date = serializers.DateTimeField()
    id_team_home = serializers.CharField(max_length=100) # models.ForeignKey(TeamsFlashscore, related_name='id_team_home', on_delete=models.PROTECT)
    id_team_away = serializers.CharField(max_length=100) # models.ForeignKey(TeamsFlashscore, related_name='id_team_away', on_delete=models.PROTECT)
    prob_home_bm =  serializers.DecimalField(max_digits=4, decimal_places=2)
    prob_draw_bm =  serializers.DecimalField(max_digits=4, decimal_places=2)
    prob_away_bm =  serializers.DecimalField(max_digits=4, decimal_places=2)
    predicted_result = serializers.CharField(max_length=5)  