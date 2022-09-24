import os

from django.shortcuts import render
from django.http import HttpResponse
from urllib.request import HTTPRedirectHandler


from scaleup import CHECKPOINTS_PATH
from scaleup.similarity import TfIdfSimilarity


# singleton
tfidf_sim = None

# Create your views here.
def input(request):
    return render(request, "input.html");

def analyse(request):
    global tfidf_sim

    resume = request.POST["resume"]
    jd1 = request.POST["jd1"]
    jd2 = request.POST["jd2"]
    jd3 = request.POST["jd3"]
    jd4 = request.POST["jd4"]
    jd5 = request.POST["jd5"]
    # print(resume)
    # print(jd1)
    # print(jd2)
    # print(jd3)
    # print(jd4)
    # print(jd5)

    # singleton
    if not tfidf_sim:
        tfidf_sim = TfIdfSimilarity(os.path.join(CHECKPOINTS_PATH, 'tfidf-1024-stopwords.joblib'))
    query = tfidf_sim.transform([resume])
    candidates = tfidf_sim.transform([jd1, jd2, jd3, jd4, jd5])
    scores = tfidf_sim.similarity(query, candidates)
    # Matching
    matches = tfidf_sim.matching(query, candidates, topk=5)
    # TODO: Not matching
    return render(request, "result.html")
