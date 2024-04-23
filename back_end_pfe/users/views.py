
from typing import Protocol
from django.contrib import messages
from django.core.mail import EmailMessage
from rest_framework.views import APIView
from rest_framework.response import Response
import jwt, datetime
from .models import User
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.exceptions import ValidationError
from django.contrib.auth.models import User
from users.models import User
from django.contrib.auth import get_user_model
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.conf import settings
from django.core.mail import send_mail
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError
from rest_framework.views import APIView
from .serializers import UserSerializer



class RegisterView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        name = request.data.get('name')

        # Check if name is more than 3 characters long
        if len(name) <= 3:
            # messages.error(request, 'Name must be more than 3 characters long')
            raise ValidationError("Name must be more than 3 characters long.")
        
        # Check if email contains '@iliadeconsulting'
        if '@iliadeconsulting' not in email:
            raise ValidationError("Email must contain '@iliadeconsulting.com'")

        # Check if password is at least 5 characters long
        if len(password) < 5:
            raise ValidationError("Password must be at least 5 characters long.")
    
        serializer = UserSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        
        activation_token = request.data.get('activation_token')
        if activation_token != User.activation_token:
            raise ValidationError('Invalid activation .')

        # Activate the user
        User.is_active = True
        User.save()
        return Response(serializer.data)
    
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import NotFound
from .models import User
class ActivateUserView(APIView):
    def get(self, request, activation_token):
        # Find the user with the corresponding activation_token
        try:
            user = User.objects.get(activation_token=activation_token, is_active=False)
        except User.DoesNotExist:
            raise NotFound("Invalid activation token or user account already activated.")

        # Activate the user account
        user.is_active = True
        user.activation_token = None  # Clear the activation token
        user.save()

        return Response({"message": "User account activated successfully."})


# ***********************************************************************************************************************************************************************************

class LoginView(APIView):
    def post(self, request):
        email = request.data['email']
        password = request.data['password']

        user = User.objects.filter(email=email).first()

        if user is None:
            raise ValidationError("User not found!")
            # raise AuthenticationFailed('User not found!')


        if not user.check_password(password):
            raise ValidationError("Incorrect password!")
            # raise AuthenticationFailed('Incorrect password!')

        # if not user.is_active:
        #     raise AuthenticationFailed('User account is not active')

        payload = {
            'id': user.id,
            'email': user.email,  # Include email in the payload

            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=60),
            'iat': datetime.datetime.utcnow()
        }

        token = jwt.encode(payload, 'secret', algorithm='HS256')

        response = Response()

        response.set_cookie(key='jwt', value=token, httponly=True)
        response.data = {
            'jwt': token,
             'user': {
                'id': user.id,
                'email': user.email,
                'role':user.role
        }}
        return response
    
#***********************************************************************************************************************************************************************************

# ***********************************************************************************************************************************************************************************

class UserView(APIView):
    def get(self, request):
        token = request.COOKIES.get('jwt')

        if not token:
            raise AuthenticationFailed('Unauthenticated!')

        try:
            payload = jwt.decode(token, 'secret', algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed('Unauthenticated!')

        user = User.objects.filter(id=payload['id']).first()
        serializer = UserSerializer(user)
        return Response(serializer.data)
    
# ***********************************************************************************************************************************************************************************

class LogoutView(APIView):
    def post(self, request):
        response = Response()
        response.delete_cookie('jwt')
        response.data = {
            'message': 'success'
        }
        return response
from django.core.mail import send_mail
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.urls import reverse
from django.conf import settings
import json
import uuid

@csrf_exempt
def send_email_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        recipient_email = data.get('projectManagerEmail', '')
        message = data.get('description', '')

        if recipient_email and message:
            subject = 'Hello chef project'
            # Get the logged-in user's email
            from_email = request.user.email if request.user.is_authenticated else 'your_email@gmail.com'
            
            # Append the request access message to the email message
            confirmation_token = str(uuid.uuid4())  # Generate a unique confirmation token
            confirmation_url = request.build_absolute_uri(reverse('confirm_access', kwargs={'token': confirmation_token}))
            message += f'\n\nRequesting access to this project. Click here to confirm: {confirmation_url}'

            try:
                send_mail(subject, message, from_email, [recipient_email])
                return JsonResponse({'message': 'Email sent successfully!'})
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

# ***********************************************************************************************************************************************************************************

from django.shortcuts import render
from django.http import HttpResponseBadRequest, HttpResponse
from django.conf import settings
from django.views.decorators.http import require_GET

@require_GET
def confirm_access(request, token):
    if token:  
        return HttpResponse('Access confirmed!')  
    else:
        return HttpResponseBadRequest('Invalid token')  

from rest_framework import generics
from rest_framework.response import Response
from rest_framework import status
from .serializers import UserSerializer

class UserListView(generics.ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer


class RegisterView(generics.CreateAPIView):
    serializer_class = UserSerializer

class UpdateUserRoleView(generics.UpdateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    lookup_url_kwarg = 'user_id'

class DeleteUserView(generics.DestroyAPIView):
    queryset = User.objects.all()
    lookup_url_kwarg = 'user_id'    