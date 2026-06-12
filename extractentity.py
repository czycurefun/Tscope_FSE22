import requests
import json
from jieba import lcut
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import Levenshtein
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import jieba
from scipy.linalg import norm
import gensim
import pandas as pd
from sklearn.metrics import cluster
from numpy import mean
from sklearn.metrics.cluster import entropy, mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import re
NMI = lambda x, y: normalized_mutual_info_score(x, y, average_method='arithmetic')

def labels_to_original(labels, original_corpus):
        assert len(labels) == len(original_corpus)
        max_label = max(labels)
        number_label = [i for i in range(0, max_label + 1, 1)]
        number_label.append(-1)
        result = [[] for i in range(len(number_label))]
        for i in range(len(labels)):
            index = number_label.index(labels[i])
            result[index].append(original_corpus[i])
        return result
def judge_contains(objone,objtwo):
    objone_list = list(objone)
    objtwo_list = list(objtwo)
    flagone = True
    flagtwo = True
    for each in objone_list:
        if each not in objtwo_list:
            flagone = False
            break
    for each in objtwo_list:
        if each not in objone_list:
            flagtwo = False
            break
    if flagone or flagtwo:
        return True
    else:
        return False



def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))
    
    s1, s2 = add_space(s1), add_space(s2)
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    numerator = np.sum(np.min(vectors, axis=0))
    denominator = np.sum(np.max(vectors, axis=0))
    return 1.0 * numerator / denominator

data_path ='./data/word2vec.vector'
with open(data_path, 'r', encoding='UTF-8') as inp_vec:
    emb_vec = inp_vec.readlines()
    word_vectors = {}
    for vec in emb_vec:
        result = vec.strip().split(' ',1)
        word_vectors[result[0]] = np.array(list((map(float, result[1].split()))))

url = "http://127.0.0.0:8000//api/EntityRelationEx/"
datalist = []

def vector_similarity(s1, s2):
        def sentence_vector(s):
            words = jieba.lcut(s)
            v = np.zeros(48)
            for word in words:
                v += word_vectors[word]
            v /= len(words)

            return v
        v1, v2 = sentence_vector(s1), sentence_vector(s2)
        return np.dot(v1, v2) / (norm(v1) * norm(v2))
testnumlist = [[4,10],[10,18],[26,32],[261,269],[326,334],[401,409],[591,598],[807,813],[868,876],[1000,1006]]
object_path = "./data/project.xlsx"
try:
    pd_sheets = pd.ExcelFile(object_path)
except Exception as e:
    print("fail".format(object_path), e)

df = pd.read_excel(pd_sheets, "Sheet1", header=[0])
temp = "最小内存余量测试"
testname = "文件/文件夹视图测试"
testcase_name = []
testcase_des = []
test_relation = []
testcasename_list = []
testcasedes_list = []
testrelation_list = []
for row in df.itertuples(index=True):
    row_list = list(row)
    testcasename = row_list[5:6]
    testdes = row_list[7:8]
    preconobject = row_list[8:9]
    test_input = row_list[9:10]
    test_expectedresult = row_list[10:11]
    testrelation = row_list[4:5]
    testitem = row_list[3:4]
    if pd.isnull(test_input):
        test_input = [""]
    if pd.isnull(test_expectedresult):
        test_expectedresult = [""]
    if pd.isnull(preconobject):
        preconobject = [""]
    if testname == testitem[0].strip() and testname != "命令":
        testcase_name.append(testcasename[0])
        testcase_des.append(testdes[0] )
        test_relation.append(testrelation[0])
    elif testname != "命令":
        testcasename_list.append(testcase_name)
        testcasedes_list.append(testcase_des)
        testrelation_list.append(test_relation)
        testcase_name = []
        testcase_des = []
        test_relation = []
        testcase_name.append(testcasename[0])
        testcase_des.append(testdes[0] )
        test_relation.append(testrelation[0])
        testname = testitem[0].strip()
    else:
        testname = testitem[0].strip()
testcasename_list.append(testcase_name)
testcasedes_list.append(testcase_des)
testrelation_list.append(test_relation)

predictlist = []
nmiscorelist = []
redundancy_TP = 0
redundancy_FP = 0
redundancy_FN = 0
unredundancy_TP = 0
unredundancy_FP = 0
unredundancy_FN = 0
kongnum = 0
log_path = "./data/result/project10.txt"
sameitemnum = 0
nosameitemnum= 0

for k,datalist in enumerate(testcasedes_list):
    resultlist = []
    for num, each in enumerate(datalist):
        if len(each) <= 200:
            datas = {"text":each}
            data = json.dumps(datas)
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, data=data, headers=headers)
            result = response.json()
            # if result[0]["entities"] != []:
            result[0]["tokens"] = each
            resultlist.append(result[0])
        else:
            each = each[0:200]
            datas = {"text":each}
            data = json.dumps(datas)
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, data=data, headers=headers)
            result = response.json()
            # if result[0]["entities"] != []:
            result[0]["tokens"] = each
            resultlist.append(result[0])

    # print(resultlist)
    entitylist = []
    for result in resultlist:
        mergeentity = ""
        testobject = ""
        testoperate = ""
        teststate = ""
        testprecon = ""
        entitydic = {}
        final_list = []
        judgeobject = True
        judgeoperate = True
        judgestate = True
        judgeprecon = True
        obj_flag = 0
        for entity in result["entities"]:
            if entity["type"] == "Component":
                obj_flag += 1
        if obj_flag >= 2:
            for entity in result["entities"]:
                if entity["type"] == "Component":
                    obj_id = entity["id"]
                    obj_value = entity["value"].replace(" ","")
                    testcasetemp_dic = {}
                    testcasetemp_dic["Component"] = obj_value
                    if result["relations"] != []:
                        for relation in result["relations"]:
                            if relation["head"].strip() != "" and relation["tail"].strip() != "":
                                if relation["tailIndex"] == obj_id:
                                    headIndex = relation["headIndex"]
                                    for each in result["entities"]:
                                        if each["id"] == headIndex:
                                            if each["type"] in testcasetemp_dic:
                                                testcasetemp_dic[each["type"]] += "," + each["value"].replace(" ","") 
                                            else:
                                                testcasetemp_dic[each["type"]] = each["value"].replace(" ","")
                    final_list.append(testcasetemp_dic)
        else:
            testcasetemp_dic = {}
            if result["entities"] != [] :
                for entity in result["entities"]:
                    testcasetemp_dic["Component"] = result["sentence"]
            else:
                if "是否支持" in result["sentence"]:
                    objentity = result["sentence"][result["sentence"].find("是否支持") + 4:len(result["sentence"])] 
                    testcasetemp_dic["Component"] = objentity
                elif  "是否可以" in result["sentence"]:
                    objentity = result["sentence"][result["sentence"].find("是否可以") + 4:len(result["sentence"])] 
                    testcasetemp_dic["Component"] = objentity
                elif "支持" in result["sentence"]:
                    objentity = result["sentence"][result["sentence"].find("支持") + 2:len(result["sentence"])] 
                    testcasetemp_dic["Component"] = objentity
                elif "可以" in result["sentence"]:
                    objentity = result["sentence"][result["sentence"].find("可以") + 2:len(result["sentence"])] 
                    testcasetemp_dic["Component"] = objentity
                else:
                    testcasetemp_dic["Component"] = result["sentence"]
            final_list.append(testcasetemp_dic)
        entitylist.append(final_list)

    
    watch_list = []
    obj_list = []
    operate_list = []
    state_list = []
    tool_list = []
    condition_list = []
    tuple_list = []
    alltuple_list = []
    num_id = 0
    for entity in entitylist:
        templist = []
        for entity_dic in entity:
            if "Component" in entity_dic:
                # tempdic = {}
                # tempdic["Component"] = entity_dic["Component"]
                # tempdic["id"] = num_id
                # watch_list.append(tempdic)
                tuple_list.append(num_id)
                obj_list.append(entity_dic["Component"].lower())
            else:
                tuple_list.append(num_id)
                obj_list.append(" ")
            if "Behavior" in entity_dic:
                tempdic = {}
                tempdic["Behavior"] = entity_dic["Behavior"]
                tempdic["id"] = num_id
                operate_list.append(entity_dic["Behavior"].lower())
            else:
                operate_list.append(" ")
            if "Prerequisite " in entity_dic:
                tempdic = {}
                tempdic["Prerequisite "] = entity_dic["Prerequisite "]
                tempdic["id"] = num_id
                state_list.append(entity_dic["Prerequisite "].lower())
            else:
                state_list.append(" ")
            if "Manner " in entity_dic:
                tempdic = {}
                tempdic["Manner "] = entity_dic["Manner "]
                tempdic["id"] = num_id
                tool_list.append(entity_dic["Manner "].lower())
            else:
                tool_list.append(" ")
            if "Constraint" in entity_dic:
                tempdic = {}
                tempdic["Constraint"] = entity_dic["Constraint"]
                tempdic["id"] = num_id
                condition_list.append(entity_dic["Constraint"].lower())
            else:
                condition_list.append(" ")
        num_id += 1
        # watch_list.append(templist)
    #w2v+sif+hdbscan
    if len(obj_list) != 1 :
        # for k,each in enumerate(testcase_name):
        #     if each == True or each == False:
        #         testcase_name[k] = ""
        text_file = " ".join([" ".join(jieba.cut(c)) for c in obj_list])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count
        all_vector_representation = np.zeros(shape=(len(obj_list), 48))
        for i, line in enumerate(obj_list):
            word_sentence = jieba.cut(line)
            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                try:
                    weight = 0.1 / (0.1 + unigram[word])
                except KeyError:
                    weight = 1
                sent_rep += wv * weight
                # sent_rep += wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

        pca = PCA(n_components=1)
        pca.fit(all_vector_representation)
        pca = pca.components_

        XXobj = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca

        all_vector_representation = np.zeros(shape=(len(operate_list), 48))
        for i, line in enumerate(operate_list):
            word_sentence = jieba.cut(line)
            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                
                sent_rep += wv 
                # sent_rep += wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep
        
        XXoperate = all_vector_representation

        
        all_vector_representation = np.zeros(shape=(len(state_list), 48))
        for i, line in enumerate(state_list):
            word_sentence = jieba.cut(line)
            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                
                sent_rep += wv 
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep
        XXstate = all_vector_representation
        all_vector_representation = np.zeros(shape=(len(tool_list), 48))
        for i, line in enumerate(tool_list):
            word_sentence = jieba.cut(line)
            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                # weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv 
                # sent_rep += wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep
        XXtool = all_vector_representation

        all_vector_representation = np.zeros(shape=(len(condition_list), 48))
        for i, line in enumerate(condition_list):
            word_sentence = jieba.cut(line)
            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                # weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv 
                # sent_rep += wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep
        XXcondition = all_vector_representation
        score_num = 0
        XXzero = np.zeros(shape=(len(obj_list), 48))
        totalsum_list = cosine_similarity(XXzero)
        objcos_simlist = cosine_similarity(XXobj)
        objcos_simlist_oral = cosine_similarity(XXobj)

        for numone,obj in enumerate(obj_list):
            for numtwo in range(numone+1,len(obj_list)):
                if judge_contains(obj_list[numone],obj_list[numtwo]) and objcos_simlist[numone][numtwo] >= 0.95:
                    objcos_simlist[numone][numtwo] = 1
                elif judge_contains(obj_list[numone],obj_list[numtwo]) and (len(obj_list[numone]) <= 4 and len(obj_list[numtwo]) <= 4):
                    objcos_simlist[numone][numtwo] = 1
                elif obj_list[numone] == obj_list[numtwo]:
                    objcos_simlist[numone][numtwo] = 1
        if not np.all(objcos_simlist == 0):
            totalsum_list += objcos_simlist
            score_num += 1
        operatecos_simlist = cosine_similarity(XXoperate)
        operatecos_simlist_oral = cosine_similarity(XXoperate)
        for numone,obj in enumerate(operate_list):
            for numtwo in range(numone+1,len(operate_list)):
                if judge_contains(operate_list[numone],operate_list[numtwo]) and operatecos_simlist[numone][numtwo] >= 0.85:
                    operatecos_simlist[numone][numtwo] = 1
        if not np.all(operatecos_simlist == 0):
            totalsum_list += operatecos_simlist
            score_num += 1
        statecos_simlist = cosine_similarity(XXstate)
        if not np.all(statecos_simlist == 0):
            totalsum_list += statecos_simlist
            score_num += 1
        toolcos_simlist = cosine_similarity(XXtool)
        if not np.all(toolcos_simlist == 0):
            totalsum_list += toolcos_simlist
            score_num += 1
        conditioncos_simlist = cosine_similarity(XXcondition)
        for numone,obj in enumerate(condition_list):
            for numtwo in range(numone+1,len(condition_list)):
                if judge_contains(condition_list[numone],condition_list[numtwo])  and conditioncos_simlist[numone][numtwo] >= 0.85:
                    conditioncos_simlist[numone][numtwo] = 1
        if not np.all(conditioncos_simlist == 0):
            totalsum_list += conditioncos_simlist
            score_num += 1
        totalsum_list = totalsum_list/score_num
        clusterlist = []
        alpha = 0.5
        threshold = 0.99
        if not (totalsum_list.min() == 1 and totalsum_list.max() == 1):
            scaler = MinMaxScaler()
            totalsum_list = scaler.fit_transform(totalsum_list)
        for num in range(2,len(totalsum_list) + 2):
            clusterlist.append(num)
        
        for numone in range(0,len(totalsum_list)):
            for numtwo in range(numone + 1,len(totalsum_list)):

                if alltuple_list[numone].keys() == alltuple_list[numtwo].keys():
                    sameitemnum += 1
                    if (len(operate_list[numone]) <= 3 and len(operate_list[numtwo]) <= 3) and (operate_list[numone] != operate_list[numtwo]):
                        continue
                    elif operate_list[numtwo] != operate_list[numone]  and len(alltuple_list[numone]) == 1:
                        continue
                    elif bool(re.search('[a-zA-Z]', obj_list[numtwo])) or bool(re.search('[a-zA-Z]', obj_list[numone])):
                        if re.sub('[\u4e00-\u9fa5]', '', obj_list[numone]) != re.sub('[\u4e00-\u9fa5]', '', obj_list[numtwo]):
                            continue
                        if bool(re.search('[a-zA-Z]', tool_list[numtwo])) or bool(re.search('[a-zA-Z]', tool_list[numone])):
                            if re.sub("[\u4e00-\u9fa5\0-9\,\。]", "", tool_list[numtwo]) != re.sub("[\u4e00-\u9fa5\0-9\,\。]", "", tool_list[numone]):
                                continue
                        elif tool_list[numtwo] != tool_list[numone]:
                            continue 
                        if condition_list[numtwo] != condition_list[numone]:
                            continue
                        if operatecos_simlist_oral[numone][numtwo] < 0.999 and operatecos_simlist_oral[numone][numtwo] !=0: 
                            continue
                        if  totalsum_list[numone][numtwo] > threshold or (obj_list[numone] == obj_list[numtwo] and len(alltuple_list[numone]) == 1 ):
                            temp = clusterlist[numone]
                            for index,each in enumerate(clusterlist):
                                if each == temp:
                                    clusterlist[index] = clusterlist[numtwo]
                    elif objcos_simlist_oral[numone][numtwo] < 0.999 and objcos_simlist_oral[numone][numtwo] !=0:
                        continue
                    else:
                        if condition_list[numtwo] != condition_list[numone]:
                            continue
                        if bool(re.search('[a-zA-Z]', tool_list[numtwo])) or bool(re.search('[a-zA-Z]', tool_list[numone])):
                            if re.sub("[\u4e00-\u9fa5\0-9\,\。]", "", tool_list[numtwo]) != re.sub("[\u4e00-\u9fa5\0-9\,\。]", "", tool_list[numone]):
                                continue
                        elif tool_list[numtwo] != tool_list[numone]:
                            continue 
                        if operatecos_simlist_oral[numone][numtwo] < 0.999 and operatecos_simlist_oral[numone][numtwo] != 0:
                            continue
                        if totalsum_list[numone][numtwo] > threshold or (obj_list[numone] == obj_list[numtwo] and len(alltuple_list[numone]) == 1 ):
                            temp = clusterlist[numone]
                            for index,each in enumerate(clusterlist):
                                if each == temp:
                                    clusterlist[index] = clusterlist[numtwo]
                else:
                    nosameitemnum += 1
        redundancy_list = []
        for each in testrelation_list[k]:
            redundancy_list.append(0)
        if len(testrelation_list[k]) == len(clusterlist):
            for numone,clusternumone in enumerate(clusterlist):
                if redundancy_list[numone] == 0:
                    for numtwo in range(numone+1,len(clusterlist)):
                        if clusternumone == clusterlist[numtwo]:
                            redundancy_list[numtwo] = 1
        else:
            new_clusterlist = []
            flag_num = 0
            for num,t in enumerate(tuple_list):
                if flag_num ==  num:
                    if (num != len(tuple_list) - 1) and (t == tuple_list[num + 1]):
                        sametuple_list = []
                        for numtwo in range(num ,len(tuple_list)):
                            if tuple_list[numtwo] == t and (numtwo != len(tuple_list) - 1):
                                sametuple_list.append(clusterlist[numtwo])
                            elif tuple_list[numtwo] == t and (numtwo == len(tuple_list) - 1):
                                sametuple_list.append(clusterlist[numtwo])
                                new_clusterlist.append(sametuple_list)
                                flag_num = len(tuple_list)
                            else:
                                new_clusterlist.append(sametuple_list)
                                flag_num = numtwo
                                break
                    else:
                        new_clusterlist.append(clusterlist[num])
                        flag_num += 1
            for cluster in new_clusterlist:
                if type(cluster) == list:
                    for num,each in enumerate(new_clusterlist):
                        if type(each) == int and (each in cluster):
                            redundancy_list[num] = 1
            for index,cluster in enumerate(new_clusterlist):
                if type(cluster) == list and (redundancy_list[index] == 0):
                    for indextwo in range(index + 1,len(new_clusterlist)):
                        if type(new_clusterlist[indextwo]) == list :
                            if (set(new_clusterlist[indextwo]) == set(cluster)) or set(new_clusterlist[indextwo]).issubset(set(cluster)):
                                redundancy_list[indextwo] = 1
            for index,cluster in enumerate(new_clusterlist):
                if type(cluster) == int and (redundancy_list[index] == 0):
                    for indextwo in range(index + 1,len(new_clusterlist)):
                        if type(new_clusterlist[indextwo]) == int:
                            if cluster == new_clusterlist[indextwo]:
                                redundancy_list[indextwo] = 1
        print(redundancy_list)

   

        for num,each in enumerate(testrelation_list[k]):
            if each == 0 and redundancy_list[num] == 0:
                unredundancy_TP += 1
            elif each == 0 and redundancy_list[num] == 1:
                unredundancy_FN += 1
                redundancy_FP += 1
            elif each == 1 and redundancy_list[num] == 1:
                redundancy_TP += 1
            elif each == 1 and redundancy_list[num] == 0:
                unredundancy_FP += 1
                redundancy_FN += 1

           


        with open(log_path, "a",encoding="utf-8") as fout:
            fout.write("********************************\n")
            fout.write("groudtruth:\n")
            for each in entitylist:
                fout.write((str(each)))
                fout.write("\n")
            fout.write(str(testrelation_list[k]))
            fout.write("\n")
            fout.write("\n")
            fout.write("predict:\n") 
            fout.write(str(redundancy_list))
            fout.write("\n")

            fout.write("\n")

redundancy_P = redundancy_TP/(redundancy_TP + redundancy_FP)
redundancy_R = redundancy_TP/(redundancy_TP + redundancy_FN)
redundancy_F1 = 2*redundancy_P*redundancy_R/(redundancy_P + redundancy_R)
unredundancy_P = unredundancy_TP/(unredundancy_TP + unredundancy_FP)
unredundancy_R = unredundancy_TP/(unredundancy_TP + unredundancy_FN)
unredundancy_F1 = 2*unredundancy_P*unredundancy_R/(unredundancy_P + unredundancy_R)
print(redundancy_TP + redundancy_FP)
print("sameitemnum: " + str(sameitemnum)) 
print("nosameitemnum: " + str(nosameitemnum)) 
print("redundancy_P: " + str(redundancy_P))
print("redundancy_R: " + str(redundancy_R))
print("redundancy_F1: " + str(redundancy_F1))
print("unredundancy_P: " + str(unredundancy_P))
print("unredundancy_R: " + str(unredundancy_R))
print("unredundancy_F1: " + str(unredundancy_F1))




