from django.db import models
from django.contrib.auth.models import AbstractUser

# At the begining of the project! 

class User(AbstractUser):
    email = models.EmailField(unique=True)
