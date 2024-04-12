from rest_framework import serializers
from .models import project_total
class project_totalSerializer(serializers.ModelSerializer):
    class Meta:
        model= project_total
        fields= ('id','name','iframeLink')

