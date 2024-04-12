import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from chatbot.chat import get_response

@csrf_exempt
def home(request):
    return render(request, 'home.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        text = data.get("message")
        if text:
            response = get_response(text)
            message = {"answer": response}
            return JsonResponse(message)
        else:
            return JsonResponse({"error": "Invalid request: No message provided."}, status=400)
    elif request.method == 'GET':
        return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)
