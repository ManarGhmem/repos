from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.tokens import default_token_generator
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.tokens import default_token_generator
# Create your models here.

class User(AbstractUser):
    name = models.CharField(max_length=255)
    email = models.CharField(max_length=255, unique=True)
    password = models.CharField(max_length=255)
    ROLE_CHOICES = (
        ('CONSULTANT', 'Consultant'),
        ('MANAGER', 'Manager'),
        ('CHEF_PROJET', 'Chef de Projet'),
    )
    role = models.CharField(max_length=255, choices=ROLE_CHOICES, default='CONSULTANT')    
    username = None

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []    


    is_active = models.BooleanField(default=False)
    activation_token = models.CharField(max_length=255, blank=True, null=True)

    def generate_activation_token(self):
        self.activation_token = default_token_generator.make_token(self)
        self.save()
        return self.activation_token