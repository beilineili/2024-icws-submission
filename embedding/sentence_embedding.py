import torch
import json
from transformers import BertModel, BertTokenizer

if __name__ == '__main__':
    local_path = "xxx/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(local_path, local_files_only=True)
    model = BertModel.from_pretrained(local_path, local_files_only=True)
    logTypeList = ["event", "auditLog"]
    
    for logType in logTypeList:
        with open("xxx/k8s_"+logType+".json", "r") as f:
            json_data = json.load(f)
        sentence_embedding = {}
        
        for key, value in json_data.items():
            sentence = ' '.join(value)
            inputs = tokenizer(sentence, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
            sentence_vector = torch.Tensor(outputs.pooler_output)
            sentence_vector_np = sentence_vector.numpy()
            sentence_embedding[sentence] = sentence_vector_np.tolist()
            
        with open("/xxx/k8s_"+logType+"_sentence_embedding.json", "w") as f:
            json.dump(sentence_embedding, f)


