from django.urls import path 
from .views import RegisterView, LoginView, UserView, LogoutView, ActivateUserView
from users import views
from . import views  # Import your views.py file
from .views import send_email_view
from .views import (
    UserListView,
    RegisterView,
    UpdateUserRoleView,
    DeleteUserView,
)
urlpatterns = [
    path('register', RegisterView.as_view()),
    path('login', LoginView.as_view()),
    path('user', UserView.as_view()),
    path('logout', LogoutView.as_view()),
    path('activate/<str:activation_token>/', ActivateUserView.as_view(), name='activate'),
    path('send-email/', send_email_view, name='send_email'),
    path('users/', views.UserListView.as_view(), name='user-list'),
    path('register/', RegisterView.as_view(), name='user-register'),
    path('users/<int:user_id>/update/', UpdateUserRoleView.as_view(), name='user-update'),
    path('users/<int:user_id>/delete/', DeleteUserView.as_view(), name='user-delete'),
    

]
