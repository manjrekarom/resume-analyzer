import os
from urllib.request import HTTPRedirectHandler

from rake_nltk import Rake
from sentence_transformers import SentenceTransformer

from django.shortcuts import render
from django.http import HttpResponse

from scaleup import CHECKPOINTS_PATH
from scaleup.keyphrase import nucleus_sampling, set_to_set_match
from scaleup.similarity import TfIdfSimilarity
from hackathon.settings import KEYPHRASE, PROJECT_RECOMMENDER, SIMILARITY, KP_LM, KP_THRESHOLD, TOP_P


# singleton
tfidf_sim = None
rake = None
sim_model = None
projects_list = None
recommender_lm = None

# Create your views here.
def input(request):
    return render(request, "input.html");


def analyse(request):
    global tfidf_sim, rake, sim_model

    if request.method == "POST":
        resume = request.FILES["resume"]
        jd1 = request.POST["jd1"]
        jd2 = request.POST["jd2"]
        jd3 = request.POST["jd3"]
        jd4 = request.POST["jd4"]
        jd5 = request.POST["jd5"]

    # similarity
    if SIMILARITY == 'TFIDF':
        # singleton
        if not tfidf_sim:
            tfidf_sim = TfIdfSimilarity(os.path.join(CHECKPOINTS_PATH, 'tfidf-1024-stopwords.joblib'))
        query = tfidf_sim.transform([str(resume.read())])
        candidates = tfidf_sim.transform([jd1, jd2, jd3, jd4, jd5])
        scores = tfidf_sim.similarity(query, candidates)
        # Matching
        matches = tfidf_sim.matching(query, candidates, topk=5)
        print('Scores and matches', scores, matches)
        # TODO: Not matching
    elif SIMILARITY == 'BM25':
        pass
    
    # handle relevant, irrelevant skills
    if KEYPHRASE == 'RAKE':
        if not rake:
            rake = Rake()
            rake.extract_keywords_from_text(resume)
            query_kp = rake.get_ranked_phrases_with_scores()
            topp_query = nucleus_sampling(query_kp, topp=TOP_P)

            for jd in [jd1, jd2, jd3, jd4, jd5]:
                if not jd:
                    continue
                rake.extract_keywords_from_text(jd)
                candidate_kp = rake.get_ranked_phrases_with_scores()
                topp_candidate = nucleus_sampling(candidate_kp, topp=TOP_P)
                
                if not sim_model:
                    sim_model = SentenceTransformer(KP_LM)
                
                rel, irrel = set_to_set_match(topp_query, topp_candidate, sim_model, 
                                              threshold=KP_THRESHOLD)
                print('Rel, irrel', rel, irrel)

    if PROJECT_RECOMMENDER == 'LM':
        pass
    
    return render(request, "result.html")
