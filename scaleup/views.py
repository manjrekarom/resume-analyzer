import os
from urllib.request import HTTPRedirectHandler

import numpy as np
from rake_nltk import Rake
from sentence_transformers import SentenceTransformer

from django.shortcuts import render
from django.http import HttpResponse

from scaleup import CHECKPOINTS_PATH
from scaleup.keyphrase import nucleus_sampling, set_to_set_match
from scaleup.similarity import LM, TfIdfSimilarity
from hackathon.settings import DATASETS_PATH, KEYPHRASE, MAX_IRREL_SKILLS, PROJECT_RECOMMENDER, PROJECTS_PER_SKILL, RECOMMENDER_LM, SIMILARITY, KP_LM, KP_THRESHOLD, TOP_P


# singleton
tfidf_sim = None
rake = None
sim_model = None
project_list = None
project_embeddings = None
recommender_lm = None

# Create your views here.
def input(request):
    return render(request, "input.html");


def analyse(request):
    global tfidf_sim, rake, sim_model, project_list,  \
        project_embeddings, recommender_lm

    if request.method == "POST":
        file = request.FILES["resume"]
        resume = str(file.read())
        jd1 = request.POST["jd1"]
        jd2 = request.POST["jd2"]
        jd3 = request.POST["jd3"]
        jd4 = request.POST["jd4"]
        jd5 = request.POST["jd5"]

    result_dict = {}
    # similarity
    if SIMILARITY == 'TFIDF':
        # singleton
        if not tfidf_sim:
            tfidf_sim = TfIdfSimilarity(os.path.join(CHECKPOINTS_PATH, 'tfidf-1024-stopwords.joblib'))
        query = tfidf_sim.transform([resume])
        candidates = tfidf_sim.transform([jd1, jd2, jd3, jd4, jd5])
        scores = tfidf_sim.similarity(query, candidates)
        # Matching
        matches = tfidf_sim.matching(query, candidates, topk=5)
        print('Scores and matches', scores, matches)
        # TODO: Not matching
        result_dict['sim_scores'] = scores
        result_dict['sim_matches'] = matches
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
                if not result_dict.get('rel', None):
                    result_dict['rel'] = []
                result_dict['rel'].append(rel)
                if not result_dict.get('irrel', None):
                    result_dict['irrel'] = []
                result_dict['irrel'].append(irrel)
    
    if result_dict.get('irrel') and PROJECT_RECOMMENDER == 'LM':
        if not project_list:
            with open(os.path.join(DATASETS_PATH, 'projects_final.list'), 'r') as f:
                project_list = f.readlines()
            project_list = np.array([project.strip() for project in project_list])
            project_embeddings = np.load(os.path.join(CHECKPOINTS_PATH, 'project_embeddings.npy')) 
        if not recommender_lm:
            recommender_lm = LM(model=SentenceTransformer(RECOMMENDER_LM))
        project_suggestions = []
        for irrel_list in result_dict['irrel']:
            irrel_kp_list = list(map(lambda x: x[0], irrel_list[:MAX_IRREL_SKILLS]))
            scores = recommender_lm.similarity(irrel_kp_list, project_embeddings)
            idxs = np.argsort(scores)[..., ::-1][..., :PROJECTS_PER_SKILL]
            project_suggestions.append(set(project_list[idxs].flatten()))
        print(project_suggestions)
    return render(request, "result.html", result_dict)
