import json
import torch
class Data_Process():
    def __init__(self):
        self.word2id = json.load(open('./word2id.json', "r"))
        self.charge2id = json.load(open('./charge2id.json'))
        self.article2id = json.load(open('./article2id.json'))
        self.time2id = json.load(open('./time2id.json'))
        self.symbol = [",", ".", "?", "\"", "”", "。", "？", "","，",",","、","”"]
        self.last_symbol = ["?", "。", "？"]
        self.charge2detail = json.load(open('./charge_details.json','r'))
        self.sent_max_len = 200
        self.law = json.load(open('./law.json'))
    def transform(self, word):
        if not (word in self.word2id.keys()):
            return self.word2id["UNK"]
        else:
            return self.word2id[word]
    
    def parse(self, sent):
        result = []
        sent = sent.strip().split()
        for word in sent:
            if len(word) == 0:
                continue
            if word in self.symbol:
                continue
            result.append(word)
        return result
    def parseH(self, sent):
        result = []
        temp = []     
        sent = sent.strip().split()
        for word in sent:
            if word in self.symbol and word not in self.last_symbol:
                continue
            temp.append(word)
            last = False
            for symbols in self.last_symbol:
                if word == symbols:
                    last = True
            if last:
                #不要标点
                result.append(temp[:-1])
                temp = []
        if len(temp) != 0:
            result.append(temp)
        
        return result

    def seq2Htensor(self, docs, max_sent=16, max_sent_len=128):
        
        sent_num_max = max([len(s) for s in docs])
        sent_num_max = min(sent_num_max, max_sent)

        sent_len_max = max([len(w) for s in docs for w in s])
        sent_len_max = min(sent_len_max, max_sent_len)
        # for lstm encoder
        #sent_tensor = torch.LongTensor(len(docs), sent_num_max, sent_len_max).zero_()
        # for textcnn encoder
        sent_tensor = torch.LongTensor(len(docs), max_sent, max_sent_len).zero_()
        
        sent_len = torch.LongTensor(len(docs), sent_num_max).zero_()
        doc_len = torch.LongTensor(len(docs)).zero_()
        for d_id, doc in enumerate(docs):
            doc_len[d_id] = len(doc)
            for s_id, sent in enumerate(doc):
                if s_id >= sent_num_max: break
                sent_len[d_id][s_id] = len(sent)
                for w_id, word in enumerate(sent):
                    if w_id >= sent_len_max: break
                    sent_tensor[d_id][s_id][w_id] = self.transform(word)
                    
        return sent_tensor

    def seq2tensor(self, sents, max_len=350):
        sent_len_max = max([len(s) for s in sents])
        sent_len_max = min(sent_len_max, max_len)

        sent_tensor = torch.LongTensor(len(sents), sent_len_max).zero_()

        sent_len = torch.LongTensor(len(sents)).zero_()
        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max: break
                #print(word)
                sent_tensor[s_id][w_id] = self.transform(word) 
        return sent_tensor,sent_len
    
    def seq2hlstm(self, docs, max_sent=16, max_sent_len=64):
        sent_num_max = max([len(s) for s in docs])
        sent_num_max = min(sent_num_max, max_sent)

        sent_len_max = max([len(w) for s in docs for w in s])
        sent_len_max = min(sent_len_max, max_sent_len)
        sent_tensor = torch.LongTensor(len(docs), sent_num_max, sent_len_max).zero_()
        sent_len = torch.LongTensor(len(docs), sent_num_max).zero_()
        doc_len = torch.LongTensor(len(docs)).zero_()
        for d_id, doc in enumerate(docs):
            doc_len[d_id] = len(doc)
            for s_id, sent in enumerate(doc):
                if s_id >= sent_num_max: break
                sent_len[d_id][s_id] = len(sent)
                for w_id, word in enumerate(sent):
                    if w_id >= sent_len_max: break
                    sent_tensor[d_id][s_id][w_id] = self.transform(word)
                    
        return sent_tensor,doc_len,sent_len
    

    def get_graph(self):
        charge_tong = json.load(open('./charge_tong.json'))
        art_tong = json.load(open('./art_tong.json'))
        charge_tong2id = {}
        id2charge_tong = {}
        legals = []
        for index,c in enumerate(charge_tong):
            charge_tong2id[c] = str(index)
            id2charge_tong[str(index)] = c
        
        legals = []  
        for i in charge_tong:
            legals.append(self.parse(self.charge2detail[i]['定义']))
           
        legals,legals_len = self.seq2tensor(legals,max_len=100)

        art2id = {}
        id2art = {}
        for index,c in enumerate(art_tong):
            art2id[c] = str(index)
            id2art[str(index)] = c
        arts = []
        for i in art_tong:
            arts.append(self.parse(self.law[str(i)]))
        arts,arts_sent_lent = self.seq2tensor(arts,max_len=150)
        
        return legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art
        

    def process_data(self,data):
        fact_all = []
        charge_label = []
        article_label = []
        time_label = []
        for index,line in enumerate(data):
            line = json.loads(line)
            fact = line['fact']
            charge = line['charge']
            article = line['article']
            if line['meta']['term_of_imprisonment']['death_penalty'] == True or line['meta']['term_of_imprisonment']['life_imprisonment'] == True:
                time_labels = 0
            else:
                time_labels = self.time2id[str(line['meta']['term_of_imprisonment']['imprisonment'])]
  
            charge_label.append(self.charge2id[charge[0]])
            article_label.append(self.article2id[str(article[0])])

            
            time_label.append(int(time_labels))

            fact_all.append(self.parse(fact))

        article_label = torch.tensor(article_label,dtype=torch.long)
        charge_label = torch.tensor(charge_label,dtype=torch.long)
        time_label = torch.tensor(time_label,dtype=torch.long)

        documents,sent_lent = self.seq2tensor(fact_all,max_len=350)
        return charge_label,article_label,time_label,documents,sent_lent


    def process_law(self,label_names,type = 'charge'):
        if type == 'charge':
            labels = []  
            for i in label_names:
                labels.append(self.parse(self.charge2detail[i]['定义']))
            labels , labels_len = self.seq2tensor(labels,max_len=100)
            return labels , labels_len
        else:
            labels = []  
            for i in label_names:
                labels.append(self.parse(self.law[str(i)]))
            labels , labels_len = self.seq2tensor(labels,max_len=150)
            return labels , labels_len