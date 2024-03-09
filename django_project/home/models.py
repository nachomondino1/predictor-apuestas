from django.db import models

# To generate a new migration: python manage.py makemigrations
# To execute the migrations: python manage.py migrate
# To see the SQL query behind the migration: python manage.py sqlmigrate <app_name> <sequence number of migration>
 
class Countries(models.Model):
   id_country = models.CharField(primary_key=True, max_length=100)
   country_name = models.CharField(max_length=255)

class Competitions(models.Model):
   id_country = models.ForeignKey(Countries, on_delete=models.PROTECT) # If we delete a country, the compretition is not deleted
   id_competition = models.CharField(primary_key=True, max_length=100)
   competition_flashscore = models.CharField(max_length=255)
   competition_sofifa = models.CharField(max_length=255)
   is_cup = models.BooleanField()


class TeamsFlashscore(models.Model):
   id_team = models.CharField(primary_key=True, max_length=100)
   team_name = models.CharField(max_length=255)
   url_team = models.URLField()


class TeamsSofifa(models.Model):
   id_team_sofifa = models.CharField(primary_key=True, max_length=100)
   team_name = models.CharField(max_length=255)
   home_stadium = models.CharField(max_length=255)
   international_prestige = models.IntegerField()
   domestic_prestige = models.IntegerField()
   id_rival_team_sofifa = models.CharField(max_length=255) # TODO: add unique param
   rival_team_name = models.CharField(max_length=255)
   url_team = models.URLField()
   id_country = models.ForeignKey(Countries, on_delete=models.PROTECT)


class CoachesFlashscore(models.Model):
   id_coach = models.CharField(primary_key=True, max_length=100)
   coach_name = models.CharField(max_length=255)
   url_coach = models.URLField()


class PlayersFlashscore(models.Model):
   id_player = models.CharField(primary_key=True, max_length=100)
   player_name = models.CharField(max_length=255)
   player_url  = models.URLField()


class PlayersSofifa(models.Model):
   id_player_sofifa = models.CharField(primary_key=True, max_length=100)
   player_name = models.CharField(max_length=255)
   height = models.IntegerField()
   preferred_foot = models.CharField(max_length=255)
   url_player = models.URLField()


class PlayersFifa(models.Model):
   id_player_sofifa = models.ForeignKey(PlayersSofifa, on_delete=models.PROTECT)
   age = models.IntegerField()
   overall_rating =  models.IntegerField()
   potential =  models.IntegerField()
   value = models.CharField(max_length=255)
   wage = models.CharField(max_length=255)
   id_team_sofifa = models.ForeignKey(TeamsSofifa, on_delete=models.PROTECT)
   fifa = models.CharField(max_length=255)
   date = models.DateTimeField()
   id_country = models.ForeignKey(Countries, on_delete=models.PROTECT)
   id_competition = models.ForeignKey(Competitions, on_delete=models.PROTECT)


class Matches(models.Model):
   id_match = models.CharField(primary_key=True, max_length=100)
   date = models.DateTimeField()
   referee = models.CharField(max_length=255)
   venue = models.CharField(max_length=255)
   capacity = models.IntegerField()
   attendance = models.IntegerField()
   goals_home = models.IntegerField()
   goals_away = models.IntegerField()
   id_team_home = models.ForeignKey(TeamsFlashscore, related_name = 'home_team', on_delete=models.PROTECT)
   id_team_away = models.ForeignKey(TeamsFlashscore, related_name = 'away_team', on_delete=models.PROTECT)
   expected_goals_xg_home = models.DecimalField(max_digits=4, decimal_places=2)
   expected_goals_xg_away = models.DecimalField(max_digits=4, decimal_places=2)
   ball_possession_home = models.CharField(max_length=3)
   ball_possession_away = models.CharField(max_length=3)
   goal_attempts_home = models.IntegerField()
   goal_attempts_away = models.IntegerField()
   shots_on_goal_home = models.IntegerField()
   shots_on_goal_away = models.IntegerField()
   shots_off_goal_home = models.IntegerField()
   shots_off_goal_away = models.IntegerField()
   free_kicks_home = models.IntegerField()
   free_kicks_away = models.IntegerField()
   corner_kicks_home = models.IntegerField()
   corner_kicks_away = models.IntegerField()
   offsides_home = models.IntegerField()
   offsides_away = models.IntegerField()
   throw_ins_home = models.IntegerField()
   throw_ins_away = models.IntegerField()
   goalkeeper_saves_home = models.IntegerField()
   goalkeeper_saves_away = models.IntegerField()
   fouls_home = models.IntegerField()
   fouls_away = models.IntegerField()
   yellow_cards_home = models.IntegerField()
   yellow_cards_away = models.IntegerField()
   total_passes_home = models.IntegerField()
   total_passes_away = models.IntegerField()
   tackles_home = models.IntegerField()
   tackles_away = models.IntegerField()
   attacks_home = models.IntegerField()
   attacks_away = models.IntegerField()
   dangerous_attacks_home = models.IntegerField()
   dangerous_attacks_away = models.IntegerField()
   clearances_completed_home = models.IntegerField()
   clearances_completed_away = models.IntegerField()
   id_coach_home = models.ForeignKey(CoachesFlashscore, related_name = 'home_team', on_delete=models.PROTECT)
   id_coach_away = models.ForeignKey(CoachesFlashscore, related_name = 'away_team', on_delete=models.PROTECT)
   id_country = models.ForeignKey(Countries, on_delete=models.PROTECT)
   id_competition = models.ForeignKey(Competitions, on_delete=models.PROTECT)
   is_cup = models.BooleanField()
   season = models.CharField(max_length=9)
   red_cards_home = models.IntegerField()
   red_cards_away = models.IntegerField()
   blocked_shots_home = models.IntegerField()
   blocked_shots_away = models.IntegerField()
   completed_passes_home = models.IntegerField()
   completed_passes_away = models.IntegerField()
   pass_success_home = models.CharField(max_length=255)
   pass_success_away = models.CharField(max_length=255)
   goal_kicks_home = models.IntegerField()
   goal_kicks_away = models.IntegerField()


class FormationFlashscore(models.Model):
 
   id_match = models.OneToOneField(Matches, primary_key=True, on_delete=models.CASCADE)

   id_player_start_home_1 = models.ForeignKey(PlayersFlashscore, related_name='start_home_1', on_delete=models.PROTECT)
   id_player_start_home_2 = models.ForeignKey(PlayersFlashscore, related_name='start_home_2', on_delete=models.PROTECT) 
   id_player_start_home_3 = models.ForeignKey(PlayersFlashscore, related_name='start_home_3', on_delete=models.PROTECT)
   id_player_start_home_4 = models.ForeignKey(PlayersFlashscore, related_name='start_home_4', on_delete=models.PROTECT)
   id_player_start_home_5 = models.ForeignKey(PlayersFlashscore, related_name='start_home_5', on_delete=models.PROTECT)
   id_player_start_home_6 = models.ForeignKey(PlayersFlashscore, related_name='start_home_6', on_delete=models.PROTECT)
   id_player_start_home_7 = models.ForeignKey(PlayersFlashscore, related_name='start_home_7', on_delete=models.PROTECT)
   id_player_start_home_8 = models.ForeignKey(PlayersFlashscore, related_name='start_home_8', on_delete=models.PROTECT)  
   id_player_start_home_9 = models.ForeignKey(PlayersFlashscore, related_name='start_home_9', on_delete=models.PROTECT)
   id_player_start_home_10 = models.ForeignKey(PlayersFlashscore, related_name='start_home_10', on_delete=models.PROTECT)  
   id_player_start_home_11 = models.ForeignKey(PlayersFlashscore, related_name='start_home_11', on_delete=models.PROTECT)

   id_player_start_away_1 = models.ForeignKey(PlayersFlashscore, related_name='start_away_1', on_delete=models.PROTECT)
   id_player_start_away_2 = models.ForeignKey(PlayersFlashscore, related_name='start_away_2', on_delete=models.PROTECT)
   id_player_start_away_3 = models.ForeignKey(PlayersFlashscore, related_name='start_away_3', on_delete=models.PROTECT) 
   id_player_start_away_4 = models.ForeignKey(PlayersFlashscore, related_name='start_away_4', on_delete=models.PROTECT) 
   id_player_start_away_5 = models.ForeignKey(PlayersFlashscore, related_name='start_away_5', on_delete=models.PROTECT)  
   id_player_start_away_6 = models.ForeignKey(PlayersFlashscore, related_name='start_away_6', on_delete=models.PROTECT) 
   id_player_start_away_7 = models.ForeignKey(PlayersFlashscore, related_name='start_away_7', on_delete=models.PROTECT)  
   id_player_start_away_8 = models.ForeignKey(PlayersFlashscore, related_name='start_away_8', on_delete=models.PROTECT)  
   id_player_start_away_9 = models.ForeignKey(PlayersFlashscore, related_name='start_away_9', on_delete=models.PROTECT) 
   id_player_start_away_10 = models.ForeignKey(PlayersFlashscore, related_name='start_away_10', on_delete=models.PROTECT)   
   id_player_start_away_11 = models.ForeignKey(PlayersFlashscore, related_name='start_away_11', on_delete=models.PROTECT)   

   id_player_sub_home_1 = models.CharField(max_length=255)
   id_player_sub_home_2 = models.CharField(max_length=255)
   id_player_sub_home_3 = models.CharField(max_length=255)
   id_player_sub_home_4 = models.CharField(max_length=255) 
   id_player_sub_home_5 = models.CharField(max_length=255)
   id_player_sub_home_6 = models.CharField(max_length=255)
   id_player_sub_home_7 = models.CharField(max_length=255)
   id_player_sub_home_8 = models.CharField(max_length=255)
   id_player_sub_home_9 = models.CharField(max_length=255) 


   id_player_sub_away_1 = models.CharField(max_length=255)
   id_player_sub_away_2 = models.CharField(max_length=255)
   id_player_sub_away_3 = models.CharField(max_length=255)
   id_player_sub_away_4 = models.CharField(max_length=255)
   id_player_sub_away_5 = models.CharField(max_length=255)
   id_player_sub_away_6 = models.CharField(max_length=255)
   id_player_sub_away_7 = models.CharField(max_length=255)
   id_player_sub_away_8 = models.CharField(max_length=255)
   id_player_sub_away_9 = models.CharField(max_length=255)


   id_player_sub_enter_home_1 = models.CharField(max_length=255)
   id_player_sub_enter_home_2 = models.CharField(max_length=255)
   id_player_sub_enter_home_3 = models.CharField(max_length=255)
   id_player_sub_enter_home_4 = models.CharField(max_length=255)
   id_player_sub_enter_home_5 = models.CharField(max_length=255)
   id_player_sub_enter_home_6 = models.CharField(max_length=255)


   id_player_sub_enter_away_1 = models.CharField(max_length=255)
   id_player_sub_enter_away_2 = models.CharField(max_length=255)
   id_player_sub_enter_away_3 = models.CharField(max_length=255)
   id_player_sub_enter_away_4 = models.CharField(max_length=255)
   id_player_sub_enter_away_5 = models.CharField(max_length=255)
   id_player_sub_enter_away_6 = models.CharField(max_length=255)


   id_player_miss_home_1 = models.CharField(max_length=255)
   id_player_miss_home_2 = models.CharField(max_length=255)
   id_player_miss_home_3 = models.CharField(max_length=255)
   id_player_miss_home_4 = models.CharField(max_length=255) 
   id_player_miss_home_5 = models.CharField(max_length=255)
   id_player_miss_home_6 = models.CharField(max_length=255)
   id_player_miss_home_7 = models.CharField(max_length=255)   
   id_player_miss_home_8 = models.CharField(max_length=255)
   id_player_miss_home_9 = models.CharField(max_length=255)
   id_player_miss_home_10 = models.CharField(max_length=255)
   id_player_miss_home_11 = models.CharField(max_length=255)
   id_player_miss_home_12 = models.CharField(max_length=255)


   id_player_miss_away_1 = models.CharField(max_length=255)
   id_player_miss_away_2 = models.CharField(max_length=255)
   id_player_miss_away_3 = models.CharField(max_length=255)
   id_player_miss_away_4 = models.CharField(max_length=255)
   id_player_miss_away_5 = models.CharField(max_length=255) 
   id_player_miss_away_6 = models.CharField(max_length=255)
   id_player_miss_away_7 = models.CharField(max_length=255)
   id_player_miss_away_8 = models.CharField(max_length=255)
   id_player_miss_away_9 = models.CharField(max_length=255) 
   id_player_miss_away_10 = models.CharField(max_length=255)  
   id_player_miss_away_11 = models.CharField(max_length=255) 
   id_player_miss_away_12 = models.CharField(max_length=255)


class Odds(models.Model):
   id_match = models.OneToOneField(Matches, on_delete=models.CASCADE, primary_key=True) # Cascade, set_null, set_default, protect
   odds_home = models.DecimalField(max_digits=5, decimal_places=2)
   odds_draw = models.DecimalField(max_digits=5, decimal_places=2)
   odds_away = models.DecimalField(max_digits=5, decimal_places=2)