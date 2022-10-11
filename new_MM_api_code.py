import numpy as np
import pickle

with open('../models/finalized_cb.sav', 'rb') as f:
    clf = pickle.load(f)

with open('../models/live_mu.sav', 'rb') as f:
    live_mu = pickle.load(f)

with open('../models/live_sigma.sav', 'rb') as f:
    live_sigma = pickle.load(f)

with open('../models/live_scores.sav', 'rb') as f:
    live_scores = pickle.load(f)

win_logic = {'0': 0.05,'1000': 0.05,'2000': 0.05,'3000': 0.05,'4000': 0.1,'5000': 0.1,'6000': 0.5}

def queue_logic(queue, times):

    queue = np.array(queue)
    
    sort = np.argsort(-1*times)
    times = times[sort]
    queue = queue[sort]

    times = np.round(times, -3).astype('int32')
    times[times > 6000] = 6000
    
    mus = []
    sigmas = []
    gscores = []

    for user_id in queue:
        try:
            mus.append(live_mu[user_id])
        except:
            mus.append(25)
        try:
            sigmas.append(live_sigma[user_id])
        except:
            sigmas.append(25/3)
        try:
            gscores.append(live_scores[user_id])
        except:
            gscores.append(500)

    mus = np.expand_dims(np.array(mus),-1)
    sigmas = np.expand_dims(np.array(sigmas),-1)
    gscores = np.expand_dims(np.array(gscores),-1)

    que = list(range(len(queue)))
    pairs = []
    no_pairs = []
    final_probs = []
    no_pair_time = []

    for i, user_id in enumerate(queue):

        if i in que:

            if len(que) > 1:

                que.remove(i)

                mu_1 = np.ones((len(que),1))*mus[i]
                sigma_1 = np.ones((len(que),1))*sigmas[i]
                gscore_1 = np.ones((len(que),1))*gscores[i]

                mu_2 = mus[que]
                sigma_2 = sigmas[que]
                gscore_2 = gscores[que]

                feats = np.concatenate((mu_1, sigma_1, mu_2, sigma_2, gscore_1, gscore_2), -1)

                probs = clf.predict_proba(feats)[:,1]
                probs1 = np.abs(probs - 0.5)

                least = np.argmin(probs1)

                if probs1[least] <= win_logic[str(times[i])]:

                    pairs.append((user_id, queue[que[least]]))
                    que.remove(que[least])
                    final_probs.append(probs[least])
                
                else:

                    no_pairs.append(user_id)
                    no_pair_time.append(times[i]+1000)

            else:
                
                no_pairs.append(user_id)
                no_pair_time.append(times[i]+1000)

    return pairs, no_pairs, no_pair_time, final_probs

def queue_logic_4p(queue, times):

    queue = np.array(queue)
    
    sort = np.argsort(-1*times)
    times = times[sort]
    queue = queue[sort]

    times = np.round(times, -3).astype('int32')
    times[times > 9000] = 9000

    mus = []
    sigmas = []
    gscores = []

    for user_id in queue:
        try:
            mus.append(live_mu[user_id])
        except:
            mus.append(25)
        try:
            sigmas.append(live_sigma[user_id])
        except:
            sigmas.append(25/3)
        try:
            gscores.append(live_scores[user_id])
        except:
            gscores.append(500)

    mus = np.expand_dims(np.array(mus),-1)
    sigmas = np.expand_dims(np.array(sigmas),-1)
    gscores = np.expand_dims(np.array(gscores),-1)

    que = list(range(len(queue)))
    pairs = []
    no_pairs = []
    final_probs = []
    no_pair_time = []
    
    for i, user_id in enumerate(queue):

        if i in que:

            if len(que) > 3:

                que.remove(i)

                mu_1 = np.ones((len(que),1))*mus[i]
                sigma_1 = np.ones((len(que),1))*sigmas[i]
                gscore_1 = np.ones((len(que),1))*gscores[i]

                mu_2 = mus[que]
                sigma_2 = sigmas[que]
                gscore_2 = gscores[que]

                feats = np.concatenate((mu_1, sigma_1, mu_2, sigma_2, gscore_1, gscore_2), -1)

                probs = clf.predict_proba(feats)[:,1]
                probs1 = np.abs(probs - 0.5)

                least = np.argsort(probs1).astype('int32')
                least = least[:3]
                
                if np.sum(probs1[least] <= win_logic[str(times[i])]) == 3:

                    pairs.append((user_id, queue[que[least[0]]], queue[que[least[1]]], queue[que[least[2]]]))
                    temp = np.array(que)[least]
                    que = [item for item in que if item not in temp]
                    final_probs.append(probs[least])
                
                else:

                    no_pairs.append(user_id)
                    no_pair_time.append(times[i]+1000)

            else:
                
                no_pairs.append(user_id)
                no_pair_time.append(times[i]+1000)

    return pairs, no_pairs, no_pair_time, final_probs
