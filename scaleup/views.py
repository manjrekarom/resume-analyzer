from urllib.request import HTTPRedirectHandler
from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def input(request):
    return render(request, "input.html");

def analyse(request):
    resume = request.POST["resume"]
    jd1 = request.POST["jd1"]
    jd2 = request.POST["jd2"]
    jd3 = request.POST["jd3"]
    jd4 = request.POST["jd4"]
    jd5 = request.POST["jd5"]
    return render(request, "result.html");
