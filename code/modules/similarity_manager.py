
import numpy as np
import sklearn.metrics.pairwise as pw


class SimilarityManager:
    def __init__(self, num_clients, n_features=20):      
        self.num_clients = num_clients
        self.num_features = n_features
        self.plain_features =  {}
        self.exist_clients =[]       

    def get_similar_clients(self, client_id, n=1, cid2 = -1, client_gate=0):      
        if client_id not in self.exist_clients:
            return None
        
        clients_sim = self.SimilarityMatrix[client_id]
        arg_sort = reversed(np.argsort(clients_sim)) # from smaller to larger value-->the bigger value at the end       
        # clients_idx_temp = arg_sort[-(n+1):]  # access the last n elements (i.e. the top n elements)
       
        clients_idx = []  
        cnt = 0
        for i in arg_sort:
            if i!=client_id and i!=cid2 and clients_sim[i] >=client_gate:
                clients_idx.append(i)
                cnt +=1
                if cnt == n:
                    break  
        # for id in clients_idx_temp:
        #     clients_sim2 = self.SimilarityMatrix[id]
        #     if np.sum(clients_sim2) != 0 and id!=client_id:
        #         clients_idx.append(id)

        if len(clients_idx) == 0:
            return None     

        return clients_idx

    def set_features_pred(self, client_id, pred_vec):
        self.features[client_id][0:28] = pred_vec
    
    def set_features_acc(self, client_id, acc_vec):
        self.features[client_id][28:30] = acc_vec

    def update_plain_features(self, client_id, features_vec):   
        if client_id not in self.exist_clients:
            self.exist_clients.append(client_id)    
        self.plain_features[client_id] = features_vec     
       
    def update_similarity_matrix(self, normalize=1):
        client_count = len(self.plain_features.items())
        plain_fet = np.zeros((client_count, self.num_features))        
        clients_id = []    
        rw = 0   
        for cid, p_fet in self.plain_features.items():
            clients_id.append(cid)
            plain_fet[rw] = p_fet
            rw +=1
        if normalize == 1:
            nor_fet = self.features_nomalization(plain_fet)
        else:
            nor_fet = plain_fet
        sm = pw.cosine_similarity(nor_fet)
        for rw in range(client_count):
            for cl in range(client_count): 
                self.SimilarityMatrix[clients_id[rw], clients_id[cl]] = sm[rw, cl]

        # self.SimilarityMatrix = pw.cosine_similarity(self.features)
    
    def features_nomalization(self, plain_fet):
        norm_features = np.zeros(plain_fet.shape)           
        for cl in range(plain_fet.shape[1]):
            cln_data = plain_fet[:,cl]
            min = np.min(cln_data)
            max = np.max(cln_data)
            if (max-min)==0:
                cln_data = 1
            else:
                cln_data = (cln_data-min)/(max-min)
            norm_features[:,cl]= cln_data
        return norm_features

    def set_features_count(self, n_features):
        self.num_features = n_features

    def similarity_manager_save(self, rnd):
        np.save("SimilarityMatrix_Exp2_rnd_"+str(rnd)+".npy", self.SimilarityMatrix)

    def print(self):
        for i in range(self.num_clients):
            print("Client_{}: Sim={}".format(i, self.SimilarityMatrix[i]))







