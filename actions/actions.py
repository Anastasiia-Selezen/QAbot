# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
import os
from typing import Any, Text, Dict, List

from haystack.document_store import ElasticsearchDocumentStore
from haystack.pipeline import ExtractiveQAPipeline
from haystack.reader import FARMReader
from haystack.retriever import DensePassageRetriever
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

elastic_host = 'localhost'
elastic_embedding_dim = 768


os.environ["TOKENIZERS_PARALLELISM"] = "False"
store = ElasticsearchDocumentStore(
    host=elastic_host,
    username="",
    password="",
    index="document",
    embedding_dim=elastic_embedding_dim,
    embedding_field="embedding")

retriever = DensePassageRetriever.load(load_dir='../QA_models/dpr',
                                       document_store=store,
                                       infer_tokenizer_classes=True,
                                       max_seq_len_query=64,
                                       max_seq_len_passage=256,
                                       batch_size=16,
                                       embed_title=True,
                                       use_fast_tokenizers=True,
                                       use_gpu=False)

reader = FARMReader(model_name_or_path='../QA_models/uk-bert-qa',
                    use_gpu=False,
                    num_processes=1)
pipe = ExtractiveQAPipeline(reader, retriever)
top_k_retriever = 5
top_k_reader = 100
top_best_answers = 5


def carousel(qa_output):
    carousel_template = {
        "type": "template",
        "payload": {
            "template_type": "generic",
            "elements": [
            ]
        }
    }

    element_template_not_found = {
        "title": "На жаль немає вірної відповіді",
        "image_url": "https://cdn3.iconfinder.com/data/icons/flat-actions-icons-9/792/Close_Icon_Dark-1024.png",
        "buttons": [{
            "title": "Повідомити",
            "url": "http://url",
            "type": "web_url"
        }]
    }

    element_template_ = {
        "title": "",
        "subtitle": "",
        "Doc url": "",
        "image_url": "https://cdn3.iconfinder.com/data/icons/flat-actions-icons-9/792/Tick_Mark_Dark-1024.png",
        "buttons": [{
            "title": "Прийняти",
            "url": "http://url",
            "type": "web_url"
        }]
    }

    for answer in qa_output:
        element_template = element_template_.copy()
        element_template["title"] = answer['answer']
        element_template["subtitle"] = "Відповідь: {} Контекст: {}".format(answer['answer'], answer['context'])
        element_template["Doc url"] = answer['meta']['name']
        carousel_template["payload"]["elements"].append(element_template)

    carousel_template["payload"]["elements"].append(element_template_not_found)

    return carousel_template


class QA(Action):
    def name(self) -> Text:
        return "action_qa"

    @staticmethod
    def qa_request(user_question: str):
        prediction = pipe.run(
            query=user_question,
            top_k_retriever=top_k_retriever,
            top_k_reader=top_k_reader)
        top_predictions = sorted(prediction['answers'], key=lambda k: k['probability'], reverse=True)[:top_best_answers]
        return top_predictions

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_question = tracker.latest_message['text']
        qa_response = self.qa_request(user_question)

        carousel_template = carousel(qa_response)
        dispatcher.utter_message(attachment=carousel_template)
        dispatcher.utter_message(text='Ще питання?')
        return []
