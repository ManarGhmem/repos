from django.urls import path
from chatbot.views import home, predict

urlpatterns = [
    path('', home, name='home'),  # Route pour la page d'accueil
    path('predict/', predict, name='predict'),  # Route pour le endpoint de prédiction
]
