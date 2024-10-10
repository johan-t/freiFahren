import grpc
import hatespeechfilter_pb2 as hatespeechfilter_pb2
import hatespeechfilter_pb2_grpc as hatespeechfilter_pb2_grpc

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import concurrent.futures as futures


class HatespeechClassifier(hatespeechfilter_pb2_grpc.HatespeechClassifierServicer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "chrisrtt/gbert-multi-class-german-hate"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "chrisrtt/gbert-multi-class-german-hate"
        )
        self.classifier = pipeline(
            "sentiment-analysis", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=1  # Use the first GPU (index 0)
        )

    def Classify(self, request, context):
        result = self.classifier(request.text)
        return hatespeechfilter_pb2.ClassificationReply(
            label=result[0]["label"], score=result[0]["score"]
        )


tokenizer = AutoTokenizer.from_pretrained("chrisrtt/gbert-multi-class-german-hate")
model = AutoModelForSequenceClassification.from_pretrained(
    "chrisrtt/gbert-multi-class-german-hate"
)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    hatespeechfilter_pb2_grpc.add_HatespeechClassifierServicer_to_server(
        HatespeechClassifier(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

serve()