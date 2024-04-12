from rest_framework import serializers
from .models import Cours

# User -> id, name, email, gender
class CoursSerializer(serializers.ModelSerializer):
    class Meta:
        model = Cours
        fields = ('id', 'titre', 'description', 'iframelink')