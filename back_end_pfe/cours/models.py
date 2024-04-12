from django.db import models

# Create your models here.
class Cours(models.Model):
    id=models.AutoField(primary_key=True)
    titre=models.CharField(max_length=255)
    description=models.CharField(max_length=255)
    iframelink=models.CharField(max_length=255)

def __str__(self):
    return self.name
