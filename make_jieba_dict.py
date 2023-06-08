import re

cnt = 0
entities = {}
words = []
labels = []
with open('./data/ner_predict_res.txt','r') as file:
    for line in file:
        if cnt%2==0:
            words = list(line.strip())
        else:
            labels = line.strip().split(' ')[1:-1]
            entity = []
            flag = 1
            for i in range(len(labels)):
                if labels[i] != 'O':
                    flag = 1
                    entity.append(words[i])
                elif flag == 1:
                    flag = 0
                    if len(entity) > 0:
                        if entities.get(''.join(entity)) == None:
                            entities[''.join(entity)] = 1
                        else:
                            entities[''.join(entity)] += 1
                        entity = []
        cnt += 1

with open('./data/jieba_dict.txt','a') as file:
    for item in entities.keys():
        file.write(f'{item} {entities[item]}\n')