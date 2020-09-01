from pyspark import SparkContext
import time,sys
from itertools import combinations

inputfile = sys.argv[1]
outputfile = sys.argv[2]

# inputfile = "/Users/peiwendu/Downloads/public_data/yelp_train.csv"
# outputfile = "peiwen_du_task1"


sc = SparkContext(appName="inf553_hw3",master="local[*]")
sc.setLogLevel("WARN")
list()

# def minHash(map):
#     user_of_business = list(map)
#     # row
#     m = len(users)
#     # hash_function
#     hashes = [[913, 901, 24593], [14, 23, 769], [1, 101, 193], [17, 91, 1543],
#               [387, 552, 98317], [11, 37, 3079], [2, 63, 97], [41, 67, 6151],
#               [91, 29, 12289], [3, 79, 53], [73, 803, 49157], [8, 119, 389]]
#     signature = dict()
#     for j,p in enumerate(hashes):
#         signature[j] = m+1
#         for i in user_of_business:
#             signature[j] = min(signature[j],(p[0]*i+p[1])%p[2]%m)
#         # print(signature[j])
#     return [signature[key] for key in signature]

# def findCandidate(map,hashes,m):
#     bid1=map[0][0]
#     bid2=map[1][0]
#     # issimilar = False
#     similarity = 0
#     if bid1!=bid2:
#         maxlen = max(len(list(map[0][1])), len(list(map[1][1])))
#         minlen = max(len(list(map[0][1])), len(list(map[1][1])))
#         if minlen >= maxlen * 0.5:
#             # character1 = minHash(list(map[0][1]))
#             # character2 = minHash(list(map[1][1]))
#             character1 = [hashfunction(list(map[0][1]),has,m) for has in hashes]
#             character2 = [hashfunction(list(map[1][1]),has,m) for has in hashes]
#             for i in range(0,12,2):
#                 if character1[i:i+2] == character2[i:i+2]:
#                     similarity = len(set(character1).intersection(set(character2))) / len(set(character1).union(set(character2)))
#                     break
#     if similarity>=0.5:
#         print(similarity)
#         # return tuple(sorted((bid1,bid2)))
#         return (tuple(sorted([bid1,bid2])),similarity)
#     else:
#         return None

def hashfunction(x,has,m):
    return min([((has[0] * u + has[1]) % has[2]) % m for u in x])

def hashtosamebin(map):
    bid = map[0]
    signatiure = map[1]
    bins = []
    for i in range(0,20,2):
        bins.append(((i/2,tuple(signatiure[i:i+2])),[bid]))
    return bins

def findSimilarity(map,b_c,buineses):
    bid1 = map[0]
    bid2 = map[1]
    character1 = set(b_c[bid1])
    character2 = set(b_c[bid2])
    # print(len(character1.intersection(character2)))
    # print(len(character1.union(character2)))
    similarity = len(character1.intersection(character2))/len(character1.union(character2))
    # print(similarity)
    return (sorted([buineses[bid1],buineses[bid2]]),similarity)


start = time.time()
test_data = sc.textFile(inputfile).filter(lambda x:"user_id, business_id, stars" not in x).map(lambda x:x.split(","))
users = test_data.map(lambda x:x[0]).distinct().collect()
businesses = test_data.map(lambda x:x[1]).distinct().collect()
# print(len(businesses))
business_index = dict()
user_index = dict()
for i,bid in enumerate(businesses):
    business_index[bid] = i
for i,uid in enumerate(users):
    user_index[uid] = i
character_matrix = test_data.map(lambda x:(business_index[x[1]],[user_index[x[0]]])).reduceByKey(lambda x, y: x+y)
characters = character_matrix.collect()
b_c = dict()
for b in characters:
    b_c[b[0]] = b[1]

m= len(users)

hashes = [[913, 51, 24593], [54, 23, 769], [622,321,6291469], [1, 101, 193], [17, 91, 1543],[765,345,3145739],
              [387, 552, 98317], [11, 37, 3079], [2, 63, 97], [41, 67, 6151], [543,342,1572869],
              [291, 29, 12289], [3, 79, 53], [473, 83, 49157], [8, 119, 389], [321,445,196613], [951,340,393241],[531,300,786433],[224,21,25165843],[987,21,12582917]]


signatiure = character_matrix.map(lambda x:(x[0],[hashfunction(x[1],has,m) for has in hashes]))

candidate_pair = signatiure.flatMap(hashtosamebin).reduceByKey(lambda x,y: x+y).filter(lambda x:len(x[1]) >1).flatMap(lambda x:list(combinations(x[1],2))).map(lambda x:tuple(sorted(x))).distinct()

similar_pair = candidate_pair.map(lambda x:findSimilarity(x,b_c,businesses)).filter(lambda x:x[1]>=0.5).sortBy(lambda x:x[0][0])
# similar_pair = candidate_pair.map(lambda x:findSimilarity(x,b_c,businesses)).filter(lambda x:x[1]>=0.5).sortBy(lambda x:x[0][0]).map(lambda x:tuple(x[0])).collect()
with open(outputfile, "w") as f:
    f.write("business_id_1, business_id_2, similarity\n")
    for pair in similar_pair.collect():
        f.write(pair[0][0]+","+pair[0][1]+","+str(pair[1])+"\n")

# end = time.time()
# print(end-start)

# no false positive, can have a few false negative, so s must be high

# vali_data = sc.textFile("/Users/peiwendu/Downloads/public_data/pure_jaccard_similarity.csv")
# head = vali_data.first()
# vali_pair = vali_data.filter(lambda x: head not in x).map(lambda x:x.split(",")).map(lambda x:(x[0],x[1])).collect()
# ture_positive = len(set(similar_pair).intersection(set(vali_pair)))
# false_positive = len(set(similar_pair).difference(set(vali_pair)))
# print(false_positive)
# false_negaive = len(set(vali_pair).difference(set(similar_pair)))
# print(false_negaive)
# precision = ture_positive/(ture_positive+false_positive)
# recall = ture_positive/(ture_positive+false_negaive)
# print(precision)
# print(recall)
