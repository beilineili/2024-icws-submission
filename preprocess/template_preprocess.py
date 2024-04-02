import pickle
import json
import nltk
from nltk.corpus import stopwords

def k8s_event_preprocess():
    event = {}
    with open("xxx/k8s_event_drain_template_miner.bin",'rb') as f:
        template_miner = pickle.load(f)
        sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
        for cluster in sorted_clusters:
            s = cluster.get_template().lower() 
            s = s.replace('<:num:>', ' ').replace('<:image:>', ' ').replace('<:*:>', ' ').replace('<:ip-address:>', ' ')
            s = s.replace(":", ' ').replace('\"', ' ').replace("\\", ' ').replace("=", ' ').replace("(", ' ').replace(")", ' ').replace("/", ' ').replace("u0000", ' ').replace(".", ' ').replace(",", ' ')
            sw_nltk = stopwords.words('english')
            words = [word for word in s.split() if word not in sw_nltk]  
            event[cluster.get_template()] = words
    return event


def k8s_auditLog_preprocess():
    auditLog = {}
    with open("xxx/k8s_auditLog_drain_template_miner.bin",'rb') as f:
        template_miner = pickle.load(f)
        sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
        for cluster in sorted_clusters:
            s = cluster.get_template().lower() 
            s = s.replace('<:num:>', ' ').replace('<:*:>', ' ')
            s = s.replace(":", ' ').replace("(", ' ').replace(")", ' ').replace("-", ' ')
            s = s.replace("serviceaccount", "service account")
            sw_nltk = stopwords.words('english')
            words = [word for word in s.split() if word not in sw_nltk]  
            auditLog[cluster.get_template()] = words
    return auditLog


if __name__ == '__main__':
    event = k8s_event_preprocess()  
    auditLog = k8s_auditLog_preprocess()
    with open("xxx/k8s_event.json", "w") as file:
        json.dump(event, file)
    with open("xxx/k8s_auditLog.json", "w") as file:
        json.dump(auditLog, file)
