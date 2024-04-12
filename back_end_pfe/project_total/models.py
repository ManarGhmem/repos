from django.db import models

# Create your models here.
class project_total(models.Model):
    id=models.AutoField(primary_key=True)
    name=models.CharField(max_length=255)
    iframeLink=models.CharField(max_length=255)
    