import grpc
import hatespeechfilter_pb2 as hatespeechfilter_pb2
import hatespeechfilter_pb2_grpc as hatespeechfilter_pb2_grpc

class HateSpeechFilterClient:
    def __init__(self, host='localhost', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = hatespeechfilter_pb2_grpc.HatespeechClassifierStub(self.channel)

    def classify(self, text):
        request = hatespeechfilter_pb2.ClassificationRequest(text=text)
        response = self.stub.Classify(request)
        return response.label, response.score

    def close(self):
        self.channel.close()

def main():
    client = HateSpeechFilterClient()
    
    try:
        # Example usage
        text_to_classify = "Fünf Kontrolleure an der S42, die sich nicht auf die Vorschriften einlassen"
        label, score = client.classify(text_to_classify)
        print(f"Text: {text_to_classify}")
        print(f"Classification: {label}")
        print(f"Confidence Score: {score}")
    finally:
        client.close()

if __name__ == "__main__":
    main()