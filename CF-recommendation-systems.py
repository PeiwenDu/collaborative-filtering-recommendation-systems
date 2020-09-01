from pyspark import SparkContext
import sys,time, numpy,math
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from itertools import combinations

# trainfile = sys.argv[1]
# testfile = sys.argv[2]
# caseid = int(sys.argv[3])
# outputfile = sys.argv[4]

trainfile = "/Users/peiwendu/Downloads/public_data/yelp_train.csv"
testfile = "/Users/peiwendu/Downloads/public_data/yelp_val.csv"
caseid = 3
outputfile = "peiwen_du_task2"

sc = SparkContext(appName="inf553_hw3.2.3", master="local[*]")
sc.setLogLevel("WARN")


if caseid==1:
    start = time.time()
    train_data = sc.textFile(trainfile).filter(lambda x:"user_id, business_id, stars" not in x).map(lambda x:x.split(","))
    test_data = sc.textFile(testfile).filter(lambda x:"user_id, business_id, stars" not in x).map(lambda x:x.split(","))
    uids1 = train_data.map(lambda x:x[0])
    bids1 = train_data.map(lambda x:x[1])
    uids2 = test_data.map(lambda x:x[0])
    bids2 = test_data.map(lambda x:x[1])
    uids = uids1.union(uids2).distinct().collect()
    bids = bids1.union(bids2).distinct().collect()
    uid_index = dict()
    bid_index= dict()
    for i,uid in enumerate(uids):
        uid_index[uid] = i
    for i,bid in enumerate(bids):
        bid_index[bid] = i
    train_ratings = train_data.map(lambda x: Rating(uid_index[x[0]], bid_index[x[1]], float(x[2])))
    test_ratings = test_data.map(lambda x: Rating(uid_index[x[0]], bid_index[x[1]], float(x[2])))

    rank = 50
    numIterations = 10
    lambdar = 0.25
    model = ALS.train(train_ratings, rank, numIterations,lambdar)

    test = test_ratings.map(lambda x:(x[0],x[1]))
    predictions = model.predictAll(test).map(lambda x:((x[0],x[1]),x[2]))
    # rate_predict = test_ratings.map(lambda x:((x[0],x[1]),x[2])).join(predictions)
    # MSE = rate_predict.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    # print("Mean Squared Error = " + str(pow(MSE,0.5)))

    results = predictions.map(lambda x:(uids[x[0][0]],bids[x[0][1]],x[1])).collect()

    with open(outputfile,"w") as f:
        f.write("user_id, business_id, prediction\n")
        for pair in results:
            f.write(pair[0]+","+pair[1]+","+str(pair[2])+"\n")
    # end = time.time()
    # print(end-start)
elif caseid == 2:
    start = time.time()
    train_data = sc.textFile(trainfile).filter(lambda x: "user_id, business_id, stars" not in x).map(
        lambda x: x.split(","))
    test_data = sc.textFile(testfile).filter(lambda x: "user_id, business_id, stars" not in x).map(
        lambda x: x.split(","))
    uids1 = train_data.map(lambda x: x[0])
    bids1 = train_data.map(lambda x: x[1])
    uids2 = test_data.map(lambda x: x[0])
    bids2 = test_data.map(lambda x: x[1])
    # print(set(uids1.collect()).difference(set(uids2.collect())))
    # print(set(uids2.collect()).difference(set(uids1.collect())))
    # print(set(bids1.collect()).difference(set(bids2.collect())))
    # print(set(bids2.collect()).difference(set(bids1.collect())))
    uids = uids1.union(uids2).distinct().collect()
    bids = bids1.union(bids2).distinct().collect()
    uid_index = dict()
    bid_index = dict()
    for i, uid in enumerate(uids):
        uid_index[uid] = i
    for i, bid in enumerate(bids):
        bid_index[bid] = i
    train_ratings = train_data.map(lambda x: (uid_index[x[0]], bid_index[x[1]], float(x[2])))
    test_ratings = test_data.map(lambda x: (uid_index[x[0]], bid_index[x[1]], float(x[2])))

    def find_rating_pair(map):
        if len(map[1]) < 2:
            return None
        for pair in combinations(map[1], 2):
            # sortedpair = sorted(list(pair),key=lambda x:x[0])
            # yield (tuple([sortedpair[0][0], sortedpair[1][0]]), [(sortedpair[0][1], sortedpair[1][1])])
            a, b = min(pair[0], pair[1]), max(pair[0], pair[1])
            yield ((a[0], b[0]), [(a[1], b[1])])

    def calculate_sim(map):
        if len(map) < 150:
            return 0
        else:
            u_rating = []
            v_rating = []
            u_sum = 0
            v_sum = 0
            for m in map:
                u_rating.append(m[0])
                u_sum += m[0]
                v_rating.append(m[1])
                v_sum += m[1]
            u_aver = u_sum / len(u_rating)
            v_aver = v_sum / len(v_rating)
            fenzi = 0
            u_fenmu = 0
            v_fenmu = 0
            for i in range(len(u_rating)):
                fenzi += (u_rating[i] - u_aver) * (v_rating[i] - v_aver)
                u_fenmu += (u_rating[i] - u_aver) ** 2
                v_fenmu += (v_rating[i] - v_aver) ** 2
            fenmu = pow(u_fenmu * v_fenmu, 0.5)
            if fenmu == 0:
                return 0
            return fenzi / fenmu

    # m = uids1.distinct().count()
    # buinesses_user_in_train = train_ratings.map(lambda x:(x[1],[(x[0],x[2])])).reduceByKey(lambda x,y:x+y).map(lambda x:(x[0],[(rate[0],math.log(m/len(x[1]))*rate[1]) for rate in x[1]]))
    buinesses_user_in_train = train_ratings.map(lambda x:(x[1],[(x[0],x[2])])).reduceByKey(lambda x,y:x+y)
    user_pairs_in_train = buinesses_user_in_train.flatMap(find_rating_pair).filter(lambda x:x!=None).reduceByKey(lambda x,y:x+y).map(lambda x:(x[0],calculate_sim(x[1]))).filter(lambda x:x[1]>0)
    user_similarity = user_pairs_in_train.flatMap(lambda x: [(x[0][0], [(x[0][1], x[1])]), (x[0][1], [(x[0][0], x[1])])]).reduceByKey(lambda x, y: x + y).collect()
    user_sim = dict()
    for u_s in user_similarity:
        user_sim[u_s[0]] = u_s[1]

    user_business_in_train = train_ratings.map(lambda x:((x[0],x[1]),x[2])).collect()
    user_business_rating = dict()
    for u_b in user_business_in_train:
        user_business_rating[u_b[0]] = u_b[1]

    user_average = train_ratings.map(lambda x: (x[0], [x[2]])).reduceByKey(lambda x, y: x + y).map(
        lambda x: (x[0], sum(x[1]) / len(x[1]))).collect()
    user_aver = dict()
    for u_a in user_average:
        user_aver[u_a[0]] = u_a[1]


    def calculate_predict(map):
        active_user = map[0]
        business = map[1]
        # if it is new item, then use rating average to replace, if it is new user, just give a initial value 3
        predict_rate = user_aver.get(active_user, 3.5)
        user_similarity = sorted(user_sim.get(active_user,[]),key=lambda x:x[1])
        # print(user_similarity)
        if len(user_similarity) == 0:
            # print("no one has similarity with this user or new user")
            return ((active_user,business),predict_rate,map[2])
        compare_user_count = 0
        fenzi = 0
        fenmu = 0
        for u_s in user_similarity:
            if compare_user_count > 10:
                break
            compare_user = u_s[0]
            simiarity = u_s[1]
            if (compare_user, business) in user_business_rating.keys():
                rui = user_business_rating[(compare_user, business)]
                compare_user_count +=1
                fenzi += (rui - user_aver[compare_user])*simiarity
                fenmu += abs(simiarity)
        # if fenmu == 0 means the similar users don't rate this item
        if fenmu != 0:
           predict_rate += fenzi/fenmu
        return ((active_user,business),predict_rate,map[2])

    result = test_ratings.map(lambda x:(x[0],x[1],x[2])).map(calculate_predict)

    RMSE = pow(result.map(lambda r: (r[1] - r[2]) ** 2).mean(),0.5)
    print(RMSE)

    # with open(outputfile, "w") as f:
    #     f.write("user_id, business_id, prediction\n")
    #     for pair in result.collect():
    #         f.write(uids[pair[0][0]] + "," + bids[pair[0][1]] + "," + str(pair[1]) + "\n")

    end = time.time()
    print(end - start)

    # # try 1
    # user_average = train_ratings.map(lambda x: (x[0], [x[2]])).reduceByKey(lambda x, y: x + y).map(lambda x: (x[0], sum(x[1]) / len(x[1]))).collect()
    # user_aver = dict()
    # for u_a in user_average:
    #     user_aver[u_a[0]] = u_a[1]
    # business_user_in_train = train_ratings.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y)
    # business_user_in_test = test_ratings.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y).collect()
    # user_business_in_train = train_ratings.map(lambda x: (x[0], [(x[1], x[2])])).reduceByKey(lambda x, y: x + y)
    # user_similarity = dict()
    # user_rate = dict()
    # for b_u in business_user_in_test:
    #     user_rate_this_buiness = business_user_in_train.filter(lambda x: x[0] == b_u[0]).map(lambda x:x[1]).flatMap(lambda x:x)
    #     for active_user in b_u[1]:
    #         business_ratings1 = user_business_in_train.filter(lambda x: x[0] == active_user).map(lambda x: x[1]).flatMap(lambda x:x)
    #         active_user_aver = user_aver[active_user]
    #         print(active_user_aver)
    #         # print(business_ratings1.collect())
    #         # print(user_rate_this_buiness.flatMap(lambda x:x))
    #         active_user_sim = []
    #         for compare_user in user_rate_this_buiness.collect():
    #             simiarity = -10
    #             if tuple(sorted([active_user,compare_user])) not in user_similarity.keys():
    #                 business_ratings2 = user_business_in_train.filter(lambda x:x[0] == compare_user).map(lambda x:x[1]).flatMap(lambda x:x)
    #                 u_v_rating = business_ratings1.join(business_ratings2).map(lambda x:(x[1][0],x[1][1]))
    #                 u_rating = u_v_rating.map(lambda x:x[0]).collect()
    #                 v_rating = u_v_rating.map(lambda x:x[1]).collect()
    #                 if len(u_rating) > 0:
    #                     u_aver = sum(u_rating) / len(u_rating)
    #                     v_aver = sum(v_rating) / len(v_rating)
    #                     u_diff = []
    #                     u_root = []
    #                     v_diff = []
    #                     v_root = []
    #                     for u in u_rating:
    #                         u_diff.append(u - u_aver)
    #                         u_root.append((u - u_aver) ** 2)
    #                     for v in v_rating:
    #                         v_diff.append(v - v_aver)
    #                         v_root.append((v - v_aver) ** 2)
    #                     fuzi = sum([u_diff[i] * v_diff[i] for i in range(len(u_rating))])
    #                     fumu = pow(sum(u_root),0.5) * pow(sum(v_root),0.5)
    #                     if fumu == 0:
    #                         simiarity = 0
    #                     else:
    #                         simiarity = fuzi / (pow(sum(u_root),0.5) * pow(sum(v_root),0.5))
    #                 print(simiarity)
    #                 user_similarity[tuple(sorted([active_user,compare_user]))] = simiarity
    #                 active_user_sim.append((compare_user,simiarity))
    #         most_similar_user = sorted(active_user_sim,key=lambda x:x[1],reverse=True)[:10]
    #         fenzi = 0
    #         fenmu = 0
    #         rating = 0
    #         for compare_user in most_similar_user:
    #             compare_user_aver = user_aver[compare_user[0]]
    #             rui = train_ratings.filter(lambda x:(x[0]==compare_user[0]) & (x[1]==b_u[0])).map(lambda x:x[2]).collect()[0]
    #             fenzi += (rui - compare_user_aver)*compare_user[1]
    #             fenmu += abs(compare_user[1])
    #         if fenmu == 0:
    #             rating = active_user_aver
    #         else:
    #             rating = active_user_aver+fenzi/fenmu
    #         user_rate[(active_user,b_u[0])] = rating
    # with open(outputfile, "w") as f:
    #     f.write("user_id, business_id, prediction\n")
    #     for pair in user_rate:
    #         f.write(uids[pair[0]] + "," + bids[pair[1]] + "," + str(user_rate[pair]) + "\n")
    #

elif caseid == 3:
    start = time.time()
    train_data = sc.textFile(trainfile).filter(lambda x: "user_id, business_id, stars" not in x).map(
        lambda x: x.split(","))
    test_data = sc.textFile(testfile).filter(lambda x: "user_id, business_id, stars" not in x).map(
        lambda x: x.split(","))
    uids1 = train_data.map(lambda x: x[0])
    bids1 = train_data.map(lambda x: x[1])
    uids2 = test_data.map(lambda x: x[0])
    bids2 = test_data.map(lambda x: x[1])
    uids = uids1.union(uids2).distinct().collect()
    bids = bids1.union(bids2).distinct().collect()
    uid_index = dict()
    bid_index = dict()
    for i, uid in enumerate(uids):
        uid_index[uid] = i
    for i, bid in enumerate(bids):
        bid_index[bid] = i
    train_ratings = train_data.map(lambda x: (uid_index[x[0]], bid_index[x[1]], float(x[2])))
    test_ratings = test_data.map(lambda x: (uid_index[x[0]], bid_index[x[1]], float(x[2])))


    def hashfunction(x, has, m):
        return min([((has[0] * u + has[1]) % has[2]) % m for u in x])


    def hashtosamebin(map):
        bid = map[0]
        signatiure = map[1]
        bins = []
        for i in range(0, 20, 2):
            bins.append(((i / 2, tuple(signatiure[i:i + 2])), [bid]))
        return bins


    def findSimilarity(map, b_c):
        bid1 = map[0]
        bid2 = map[1]
        character1 = set(b_c[bid1])
        character2 = set(b_c[bid2])
        # print(len(character1.intersection(character2)))
        # print(len(character1.union(character2)))
        similarity = len(character1.intersection(character2)) / len(character1.union(character2))
        # print(similarity)
        return (tuple(sorted([bid1, bid2])), similarity)


    uid_in_train = uids1.distinct().collect()
    uid_index_in_train = dict()
    for i_u in enumerate(uid_in_train):
        uid_index_in_train[i_u[1]] = i_u[0]
    m = len(uid_index_in_train)

    character_matrix = train_data.map(lambda x: (uid_index_in_train[x[0]], bid_index[x[1]], float(x[2]))).map(lambda x:(x[1],[x[0]])).reduceByKey(lambda x,y:x+y)
    # character_matrix = train_ratings.map(lambda x:(x[1],[x[0]])).reduceByKey(lambda x,y:x+y)
    characters = character_matrix.collect()
    b_c = dict()
    for b in characters:
        b_c[b[0]] = b[1]


    # m = uids1.distinct().count()

    hashes = [[913, 51, 24593], [54, 23, 769], [622, 321, 6291469], [1, 101, 193], [17, 91, 1543], [765, 345, 3145739],
              [387, 552, 98317], [11, 37, 3079], [2, 63, 97], [41, 67, 6151], [543, 342, 1572869],
              [291, 29, 12289], [3, 79, 53], [473, 83, 49157], [8, 119, 389], [321, 445, 196613], [951, 340, 393241],
              [531, 300, 786433], [224, 21, 25165843], [987, 21, 12582917]]

    signatiure = character_matrix.map(lambda x: (x[0], [hashfunction(x[1], has, m) for has in hashes]))

    candidate_pair = signatiure.flatMap(hashtosamebin).reduceByKey(lambda x, y: x + y).filter(
        lambda x: len(x[1]) > 1).flatMap(lambda x: list(combinations(x[1], 2))).map(
        lambda x: tuple(sorted(x))).distinct()

    similar_pair = candidate_pair.map(lambda x: findSimilarity(x, b_c)).filter(
        lambda x: x[1] >= 0.5)

    # print(similar_pair.take(5))
    def find_rating_pair(map):
        if len(map[1]) < 2:
            return None
        for pair in combinations(map[1], 2):
            # sortedpair = sorted(list(pair),key=lambda x:x[0])
            # yield (tuple([sortedpair[0][0], sortedpair[1][0]]), [(sortedpair[0][1], sortedpair[1][1])])
            a, b = min(pair[0], pair[1]), max(pair[0], pair[1])
            yield ((a[0], b[0]), [(a[1], b[1])])

    def calculate_sim(map):
        if len(map) < 500:
            return 0
        else:
            u_rating = []
            v_rating = []
            u_sum = 0
            v_sum = 0
            for m in map:
                u_rating.append(m[0])
                u_sum += m[0]
                v_rating.append(m[1])
                v_sum += m[1]
            u_aver = u_sum / len(u_rating)
            v_aver = v_sum / len(v_rating)
            fenzi = 0
            u_fenmu = 0
            v_fenmu = 0
            for i in range(len(u_rating)):
                fenzi += (u_rating[i] - u_aver) * (v_rating[i] - v_aver)
                u_fenmu += (u_rating[i] - u_aver) ** 2
                v_fenmu += (v_rating[i] - v_aver) ** 2
            fenmu = pow(u_fenmu * v_fenmu, 0.5)
            if fenmu == 0:
                return 0
            # print(fenzi/fenmu)
            return fenzi / fenmu


    user_business_in_train = train_ratings.map(lambda x:((x[0]),[(x[1],x[2])])).reduceByKey(lambda x,y:x+y)
    item_pairs_in_train = user_business_in_train.flatMap(find_rating_pair).filter(lambda x:x!=None).reduceByKey(lambda x,y:x+y)
    item_pairs = item_pairs_in_train.join(similar_pair).map(lambda x:(x[0],calculate_sim(x[1][0]))).filter(lambda x:x[1]>0)
    # item_pairs = item_pairs_in_train.map(lambda x:(x[0],calculate_sim(x[1])))
    # item_similarity = item_pairs.flatMap(lambda x: [(x[0][0], [(x[0][1], x[1])]), (x[0][1], [(x[0][0], x[1])])]).reduceByKey(lambda x, y: x+y).collect()
    item_similarity = item_pairs.collect()
    item_sim = dict()
    for i_s in item_similarity:
        item_sim[i_s[0]] = i_s[1]

    all_other_rated_item = dict()
    for u_i in user_business_in_train.collect():
        all_other_rated_item[u_i[0]] = u_i[1]

    def calculate_predict(map):
        user = map[0]
        active_item = map[1]
        predict_rating = 3.5
        fenzi = 0
        fenmu = 0
        other_rated_items = all_other_rated_item.get(user,[])
        if len(other_rated_items) == 0:
            # print("user only rates one or new item")
            return ((user, active_item), predict_rating,map[2])
        # print(other_rated_items)
        m = 1/len(other_rated_items)
        for compare_item in other_rated_items:
            item_pair = tuple(sorted([active_item,compare_item[0]]))
            # print(item_sim.get(item_pair, 0))
            fenzi += compare_item[1]*item_sim.get(item_pair,m)
            # print(fenzi)
            fenmu+=abs(item_sim.get(item_pair,m))
        if fenmu !=0:
            predict_rating = fenzi / fenmu
        return ((user, active_item), predict_rating,map[2])


    # user_business = train_ratings.map(lambda x: ((x[0], x[1]), x[2])).collect()
    # user_business_rating = dict()
    # for u_b in user_business:
    #     user_business_rating[u_b[0]] = u_b[1]
    #
    # user_average = train_ratings.map(lambda x: (x[0], [x[2]])).reduceByKey(lambda x, y: x + y).map(
    #     lambda x: (x[0], sum(x[1]) / len(x[1]))).collect()
    # user_aver = dict()
    # for u_a in user_average:
    #     user_aver[u_a[0]] = u_a[1]
    #
    # def calculate_predict(map):
    #     user = map[0]
    #     active_item = map[1]
    #     predict_rating = user_aver.get(user,3.5)
    #     similar_items = sorted(item_sim.get(active_item,[]),key=lambda x:x[1])
    #     if len(similar_items) == 0:
    #         # print("no one has similarity with this item or new item")
    #         return ((user,active_item),predict_rating,map[2])
    #     compare_item_count = 0
    #     fenzi = 0
    #     fenmu = 0
    #     for i_s in similar_items:
    #         if compare_item_count > 10:
    #             break
    #         compare_item = i_s[0]
    #         simiarity = i_s[1]
    #         if (user, compare_item) in user_business_rating.keys():
    #             fenzi+=user_business_rating[(user,compare_item)]*simiarity
    #             fenmu+=abs(simiarity)
    #     if fenmu != 0:
    #         predict_rating = fenzi / fenmu
    #     return ((user, active_item), predict_rating,map[2])

    result = test_ratings.map(calculate_predict)

    RMSE = pow(result.map(lambda r: (r[1] - r[2]) ** 2).mean(),0.5)
    print(RMSE)
    #
    # with open(outputfile, "w") as f:
    #     f.write("user_id, business_id, prediction\n")
    #     for pair in result.collect():
    #         f.write(uids[pair[0][0]] + "," + bids[pair[0][1]] + "," + str(pair[1]) + "\n")

    end = time.time()
    print(end - start)