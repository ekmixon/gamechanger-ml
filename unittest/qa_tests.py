import requests
import logging
import json
import time

logger = logging.getLogger()
API_URL = "http://localhost:5000/questionAnswer"

def send_qa(query, context):
    
    start = time.perf_counter()
    post = {
        "query": query,
        "search_context": context
    }
    data = json.dumps(post).encode("utf-8")
    headers = {'Content-Type': 'application/json'}
    response = requests.post(API_URL, data = data, headers = headers)

    end = time.perf_counter()
    took = float(f"{end-start:0.4f}")

    return response.json(), took
    
qa_test_context_1 = [
        "Virginia'\''s Democratic-controlled Legislature passed a bill legalizing the possession of small amounts of marijuana on Wednesday, making it the 16th state to take the step. Under Virginia'\''s new law, adults ages 21 and over can possess an ounce or less of marijuana beginning on July 1, rather than Jan. 1, 2024. Gov. Ralph Northam, a Democrat, proposed moving up the date, arguing it would be a mistake to continue to penalize people for possessing a drug that would soon be legal. Lt. Gov. Justin Fairfax, also a Democrat, broke a 20-20 vote tie in Virginia'\''s Senate to pass the bill. No Republicans supported the measure. Democratic House of Delegates Speaker Eileen Filler-Corn hailed the plan. Today, with the Governor'\''s amendments, we will have made tremendous progress in ending the targeting of Black and brown Virginians through selective enforcement of marijuana prohibition by this summer she said in a statement. Republicans voiced a number of objections to what they characterized as an unwieldy, nearly 300-page bill. Several criticized measures that would grant licensing preferences to people and groups who'\''ve been affected by the war on drugs and make it easier for workers in the industry to unionize. Senate Minority Leader Tommy Norment also questioned Northam'\''s motives.",
        "We have a governor who wants to contribute to the resurrection of his legacy, Norment said, referring to the 2019 discovery of a racist photo in Northam'\''s 1984 medical school yearbook. The accelerated timeline sets Virginia cannabis consumers in an unusual predicament. While it will be legal to grow up to four marijuana plants beginning July 1, it could be several years before the state begins licensing recreational marijuana retailers. And unlike other states, the law won'\''t allow the commonwealth'\''s existing medical dispensaries to begin selling to all adults immediately. Jenn Michelle Pedini, executive director of Virginia NORML, called legalization an incredible victory but said the group would continue to push to allow retail sales to begin sooner.",
        "In the interest of public and consumer safety, Virginians 21 and older should be able to purchase retail cannabis products at the already operational dispensaries in 2021, not in 2024, Pedini said in a statement. Such a delay will only exacerbate the divide for equity applicants and embolden illicit activity. Northam and other Democrats pitched marijuana legalization as a way to address the historic harms of the war on drugs. One state study found Black Virginians were 3.5 times more likely to be arrested on marijuana charges compared with white people. Those trends persisted even after Virginia reduced penalties for possession to a $25 civil fine. New York and New Jersey also focused on addressing those patterns when governors in those states signed laws to legalize recreational marijuana this year. Northam'\''s proposal sets aside 30% of funds to go to communities affected by the war on drugs, compared with 70% in New Jersey. Another 40% of Virginia'\''s revenue will go toward early childhood education, with the remainder funding public health programs and substance abuse treatment.",
        "Those plans, and much of the bill'\''s regulatory framework, are still tentative; Virginia lawmakers will have to approve them again during their general session next year. Some criminal justice advocates say lawmakers should also revisit language that creates a penalty for driving with an open container of marijuana. In the absence of retail sales, some members of law enforcement said it'\''s not clear what a container of marijuana will be. The bill specifies a category of social equity applicants, such as people who'\''ve been charged with marijuana-related offenses or who graduated from historically Black colleges and universities. Those entrepreneurs will be given preference when the state grants licensing. Mike Thomas, a Black hemp cultivator based in Richmond who served jail time for marijuana possession, said those entrepreneurs deserved special attention. Thomas said he looked forward to offering his own line of organic, craft cannabis. Being that the arrest rate wasn'\''t the same for everyone, I don'\''t think the business opportunities should be the same for everyone"
    ]
    
def test_regular():
    query = "when is marijuana legalized"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < 10 # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_one_question():
    query = "when is marijuana legalized?"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < 10 # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_multiple_question():
    query = "when is marijuana legalized???"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < 10 # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_allcaps():
    query = "WHEN IS MARIJUANA LEGALIZED"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < 10 # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_apostrophe():
    query = "when's marijuana legalized"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < 10 # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_past_tense():
    query = "when was marijuana legalized?"
    expected = 'Wednesday'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < 10 # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_future_tense():
    query = "when will marijuana be legal?"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < 10 # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_specific():
    query = "when will marijuana be legal in Virginia?"
    expected = 'July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < 10 # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_outside_scope():
    query = "what is the capital of Assyria?"
    expected = ''
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < 10 # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer