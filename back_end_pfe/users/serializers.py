from rest_framework import serializers
from .models import User
from django.core.mail import send_mail

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'name', 'email', 'password', 'role']
        extra_kwargs = {
            'password': {'write_only': True}
        }

    def create(self, validated_data):
        password = validated_data.pop('password', None)
        instance = self.Meta.model(**validated_data)
        if password is not None:
            instance.set_password(password)
        # instance.save()
        instance.is_active = False
        instance.save()
        # Generate and send email verification token
        activation_token = instance.generate_activation_token()
        send_activation_email(instance.email, activation_token)
        
        return instance

def send_activation_email(email, activation_token):
    subject = 'Activate Your Account'
    message = f'Your activation token is: {activation_token}'
    send_mail(subject, message, 'ghmemmanar05@gmail.com', [email])
       
        # # Generate and send email verification token
        # activation_token = instance.generate_activation_token()
        # send_verification_email(instance.email, activation_token)
        