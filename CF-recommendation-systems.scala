
import org.apache.spark.{SparkConf, SparkContext,rdd}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import breeze.numerics.{abs, pow}
import java.io.PrintWriter


object peiwen_du_task2 {
  def main(args: Array[String]): Unit = {
    val trainfile =args(0)
    val testfile = args(1)
    val caseid = args(2).toInt
    val outputfile = args(3)
//    println(trainfile)
//    println(testfile)
//    println(caseid)
//    println(outputfile)

//    val trainfile = "/Users/peiwendu/Downloads/public_data/yelp_train.csv"
//    val testfile = "/Users/peiwendu/Downloads/public_data/yelp_val.csv"
//    val caseid = 3
//    val outputfile = "peiwen_du_task2"

    val conf = new SparkConf()
    conf.setAppName("inf553_hw3.2")
    conf.setMaster("local[*]")
    val sc = new SparkContext(conf)

    if (caseid==1){
//      val start = System.currentTimeMillis()
      val train_data = sc.textFile(trainfile).filter(_!="user_id, business_id, stars").map(x=>x.split(","))
      val test_data =  sc.textFile(testfile).filter(_!="user_id, business_id, stars").map(x=>x.split(","))
      val uids1 = train_data.map(_(0))
      val bids1 = train_data.map(_(1))
      val uids2 = test_data.map(_(0))
      val bids2 = test_data.map(_(1))
      val uids = uids1.union(uids2).distinct().collect()
      val bids = bids1.union(bids2).distinct().collect()
      val uid_index = scala.collection.mutable.HashMap.empty[String, Int]
      val bid_index= scala.collection.mutable.HashMap.empty[String, Int]
      val m1 = uids.length
      val m2 = bids.length
      for (i<-0 until m1 ){
        uid_index(uids(i)) = i
//        println(uids(i))
      }
      for (j<-0 until m2){
        bid_index(bids(j)) = j
      }
      val train_ratings = train_data.map(x=> Rating(uid_index(x(0)), bid_index(x(1)), x(2).toFloat))
      val test_ratings = test_data.map(x=> Rating(uid_index(x(0)), bid_index(x(1)),  x(2).toFloat))

      val rank = 50
      val numIterations = 10
      val lambdar = 0.25
      val model = ALS.train(train_ratings, rank, numIterations,lambdar)

      val test = test_ratings.map(x=>(x.user,x.product))
//      val predictions = model.predictAll(test).map(lambda x:((x[0],x[1]),x[2]))
      val predictions = model.predict(test).map(x=>((x.user,x.product),x.rating))
//      val rate_predict = test_ratings.map(x=>((x.user,x.product),x.rating)).join(predictions)
//      val RMSE = pow(rate_predict.map(r=> pow(r._2._1-r._2._2,2)).mean(),0.5)
//      print("Root Mean Squared Error = ")
//      println(RMSE)
//      for (i<-train_data.take(2)){
//        println(i(0))
//        println(i(1))
//      }

//      val results = predictions.map(x=>(uids(x._1._1),bids(x._1._2),x._2)).collect()

      val pw = new PrintWriter(outputfile)
      pw.println("user_id, business_id, prediction")
      for (pair <- predictions.collect())
        pw.println(uids(pair._1._1)+","+bids(pair._1._2)+","+pair._2.toString)
      pw.close()
//      val end = System.currentTimeMillis()
//      println(end-start)

    }else if (caseid==2){
      val start = System.currentTimeMillis()
      val train_data = sc.textFile(trainfile).filter(_!="user_id, business_id, stars").map(x=>x.split(","))
      val test_data =  sc.textFile(testfile).filter(_!="user_id, business_id, stars").map(x=>x.split(","))
      val uids1 = train_data.map(_(0))
      val bids1 = train_data.map(_(1))
      val uids2 = test_data.map(_(0))
      val bids2 = test_data.map(_(1))
      val uids = uids1.union(uids2).distinct().collect()
      val bids = bids1.union(bids2).distinct().collect()
      val uid_index = scala.collection.mutable.HashMap.empty[String, Int]
      val bid_index= scala.collection.mutable.HashMap.empty[String, Int]
      val m1 = uids.length
      val m2 = bids.length
      for (i<-0 until m1 by 1){
        uid_index(uids(i)) = i
        //        println(uids(i))
      }
      for (i<-0 until m2 by 1){
        bid_index(bids(i)) = i
      }
      val train_ratings = train_data.map(x=> (uid_index(x(0)), bid_index(x(1)), x(2).toFloat))
      val test_ratings = test_data.map(x=> (uid_index(x(0)), bid_index(x(1)),  x(2).toFloat))

      val buinesses_user_in_train = train_ratings.map(x=> (x._2,List((x._2,x._3)))).reduceByKey((x,y)=>List.concat(x, y))

      val user_pairs_in_train = buinesses_user_in_train.flatMap(x=>{
//          val maplength = x._2.length
//          if(maplength<2){
//            return None
//          }
          val combinepair = scala.collection.mutable.ListBuffer.empty[Tuple2[Tuple2[Int,Int],List[Tuple2[Float,Float]]]]
          for (y <- x._2.combinations(2)) {
            val temp = scala.collection.mutable.ListBuffer.empty[(Float, Float)]
            var a = (0, 0.toFloat)
            var b = (0, 0.toFloat)
            if (y(0)._1 > y(1)._1) {
              a = y(1)
              b = y(0)
            }
            else {
              a = y(0)
              b = y(1)
            }
            temp += Tuple2(a._2, b._2)
            combinepair += Tuple2((a._1, b._1), temp.toList)
          }

        combinepair
      }).reduceByKey((x,y)=>List.concat(x,y)).map(x=>{
        var similarity = 0.0
        if(x._2.length>200) {
          var u_rating = scala.collection.mutable.ListBuffer.empty[Float]
          var v_rating = scala.collection.mutable.ListBuffer.empty[Float]
          var u_sum = 0.0
          var v_sum = 0.0
          for (m <- x._2) {
            u_rating += m._1
            u_sum += m._1
            v_rating += m._2
            v_sum += m._2
          }
          val length_of_rating = u_rating.length
          val u_aver = u_sum / length_of_rating
          val v_aver = v_sum / length_of_rating
          var fenzi = 0.0
          var u_fenmu = 0.0
          var v_fenmu = 0.0

          for (i <- 0 until length_of_rating by 1) {
            fenzi += (u_rating(i) - u_aver) * (v_rating(i) - v_aver)
            u_fenmu += pow(u_rating(i) - u_aver, 2)
            v_fenmu += pow(v_rating(i) - v_aver, 2)
          }
          val fenmu = pow(u_fenmu * v_fenmu, 0.5)
          if (fenmu != 0) {
            similarity = fenzi / fenmu
          }
        }
//        print(similarity)
        (x._1,similarity)
      }).filter(_._2>0)
      val user_similarity = user_pairs_in_train.flatMap(x=>List(Tuple2(x._1._1,List(Tuple2(x._1._2,x._2))), Tuple2(x._1._2,List(Tuple2(x._1._1,x._2))))).reduceByKey((x, y)=>List.concat(x,y)).collect()
      val user_sim = scala.collection.mutable.HashMap.empty[Int,List[Tuple2[Int,Double]]]
      for (u_s<-user_similarity){
        user_sim(u_s._1) = u_s._2
      }

      val user_business_in_train = train_ratings.map(x=>((x._1,x._2),x._3)).collect()
      val user_business_rating = scala.collection.mutable.HashMap.empty[Tuple2[Int,Int],Float]
      for (u_b<- user_business_in_train){
        user_business_rating(u_b._1) = u_b._2
      }

      val user_average = train_ratings.map(x=> (x._1, List(x._3))).reduceByKey((x,y)=> List.concat(x,y)).map(
        x=>(x._1, x._2.sum / x._2.length)).collect()
      val user_aver = scala.collection.mutable.HashMap.empty[Int,Float]
      for (u_a <- user_average){
        user_aver(u_a._1) = u_a._2
      }

      val result = test_ratings.map(x=> {
        val active_user = x._1
        val business = x._2
        var predict_rate = 3.5
        if (user_aver.contains(active_user)){
          predict_rate = user_aver(active_user).toDouble
        }
        val user_similarity = user_sim.getOrElse(active_user,List.empty).sortBy(x=>x._2)
//        println(user_similarity)
        if (user_similarity.nonEmpty){
          var compare_user_count = 0
          var fenzi = 0.0
          var fenmu = 0.0
          for (u_s<-user_similarity) {
            if (compare_user_count <= 10) {
              val compare_user = u_s._1
              var similarity = u_s._2
              if (user_business_rating.contains(Tuple2(compare_user, business))){
                val rui = user_business_rating(Tuple2(compare_user, business))
                compare_user_count += 1
                fenzi = fenzi + (rui - user_aver(compare_user)) * similarity
                fenmu += abs(similarity)
              }
            }
          }
          if (fenmu != 0){
//            println(fenzi/fenmu)
            predict_rate += fenzi/fenmu
          }
        }
        ((active_user,business),predict_rate,x._3)
      })

//      val RMSE = pow(result.map(x=>pow(x._2 - x._3,2)).mean(),0.5)
//      print(RMSE)

      val pw = new PrintWriter(outputfile)
      pw.println("user_id, business_id, prediction")
      for (pair <- result.collect())
        pw.println(uids(pair._1._1)+","+bids(pair._1._2)+","+pair._2.toString)
      pw.close()
//      val end = System.currentTimeMillis()
//      println(end-start)

    }else if (caseid==3){
      val start = System.currentTimeMillis()
      val train_data = sc.textFile(trainfile).filter(_!="user_id, business_id, stars").map(x=>x.split(","))
      val test_data =  sc.textFile(testfile).filter(_!="user_id, business_id, stars").map(x=>x.split(","))
      val uids1 = train_data.map(_(0))
      val bids1 = train_data.map(_(1))
      val uids2 = test_data.map(_(0))
      val bids2 = test_data.map(_(1))
      val uids = uids1.union(uids2).distinct().collect()
      val bids = bids1.union(bids2).distinct().collect()
      val uid_index = scala.collection.mutable.HashMap.empty[String, Int]
      val bid_index= scala.collection.mutable.HashMap.empty[String, Int]
      val m1 = uids.length
      val m2 = bids.length
      for (i<-0 until m1 by 1){
        uid_index(uids(i)) = i
      }
      for (i<-0 until m2 by 1){
        bid_index(bids(i)) = i
      }
      val train_ratings = train_data.map(x=> (uid_index(x(0)), bid_index(x(1)), x(2).toFloat))
      val test_ratings = test_data.map(x=> (uid_index(x(0)), bid_index(x(1)),  x(2).toFloat))

      val user_in_train = uids1.collect()
      val m = user_in_train.length
      val user_index_in_train = scala.collection.mutable.HashMap.empty[String,Int]
      for (i <- 0 until m ){
        user_index_in_train(user_in_train(i)) = i
      }

      val character_matrix = test_data.map(x=>(bid_index(x(1)),user_index_in_train(x(0)))).groupByKey().mapValues(_.toList)
      val characters = character_matrix.collect()

      val b_c = scala.collection.mutable.HashMap.empty[Int, List[Int]]
      for (b<-characters){
        b_c(b._1) = b._2
      }

      val hashes = List(List(913, 51, 24593), List(2, 59, 53),List(232,45,201326611), List(54, 23, 769), List(622,321,6291469), List(1, 101, 193),List(431,21,805306457), List(17, 91, 1543),List(765,345,3145739),
        List(387, 552, 98317), List(11, 37, 3079),List(2123,222,100663319), List(2, 63, 97), List(543,342,1572869),List(145,234,25165843), List(41, 67, 6151),List(333,112,50331653),List(333,21,1610612741),
        List(291, 29, 12289), List(3, 79, 53),List(432,14,402653189), List(321,445,196613), List(951,340,393241),List(531,300,786433), List(473, 83, 49157),List(224,21,25165843), List(8, 119, 389),List(987,21,12582917))

      val signatiure = character_matrix.map(x=>{
        val tmp = scala.collection.mutable.ListBuffer.empty[Int]
        for (has <-hashes){
          var min = m+1
          for (u<-x._2){
            var hashvalue = ((has(0) *u +has(1))%has(2)) % m
            if (min >= hashvalue){
              min = hashvalue
            }
          }
          tmp += min
        }
        Tuple2(x._1,tmp.toList)
      })

      val candidate_pair = signatiure.flatMap(x=>{
        val bid = x._1
        val signature = x._2
        var bins = scala.collection.mutable.ListBuffer.empty[((Int, Tuple1[List[Int]]), List[Int])]
        for (i<- 0 until 28 by 2){
          bins += Tuple2(Tuple2(i/2, Tuple1(signature.slice(i, i+2))), List(bid))
        }
        bins.toList
      }).reduceByKey((x,y)=>List.concat(x, y)).filter(_._2.length >1).flatMap(x=> {
        val res = scala.collection.mutable.ListBuffer.empty[((Int, Int), Int)]
        val length = x._2.length
        val whole = x._2.sorted
        for (i <- 0 until length) {
          for (j <- i+1 until length) {
            res += Tuple2(Tuple2(whole(i), whole(j)), 1)
          }
        }
        res.toList
      }).reduceByKey((x, y) => x).map(x => x._1)

      val similar_pair = candidate_pair.map(x=> {
        val bid1 = x._1
        val bid2 = x._2
        val character1 = b_c(bid1).toSet
        val character2 = b_c(bid2).toSet
        val similarity = character1.intersect(character2).size.toFloat / character1.union(character2).size
        val pair = List(bid1,bid2).sorted
        val first = pair(0)
        val second = pair(1)
        Tuple2(Tuple2(first, second), similarity)
      }).filter(_._2>=0.5)

      val user_business_in_train = train_ratings.map(x=>(x._1,List((x._2,x._3)))).reduceByKey((x,y)=>List.concat(x,y))
      val item_pairs_in_train = user_business_in_train.flatMap(x=>{
        val combinepair = scala.collection.mutable.ListBuffer.empty[Tuple2[Tuple2[Int,Int],List[Tuple2[Float,Float]]]]
        for (y <- x._2.combinations(2)) {
          val temp = scala.collection.mutable.ListBuffer.empty[(Float, Float)]
          var a = (0, 0.toFloat)
          var b = (0, 0.toFloat)
          if (y(0)._1 > y(1)._1) {
            a = y(1)
            b = y(0)
          }
          else {
            a = y(0)
            b = y(1)
          }
          temp += Tuple2(a._2, b._2)
          combinepair += Tuple2((a._1, b._1), temp.toList)
        }
        combinepair
      }).reduceByKey((x,y)=>List.concat(x,y))

      val item_pairs = item_pairs_in_train.join(similar_pair).map(x=>{
        var similarity = 0.0
        if(x._2._1.length > 500) {
          var u_rating = scala.collection.mutable.ListBuffer.empty[Float]
          var v_rating = scala.collection.mutable.ListBuffer.empty[Float]
          var u_sum = 0.0
          var v_sum = 0.0
          for (m <- x._2._1) {
            u_rating += m._1
            u_sum += m._1
            v_rating += m._2
            v_sum += m._2
          }
          val length_of_rating = u_rating.length
          val u_aver = u_sum / length_of_rating
          val v_aver = v_sum / length_of_rating
          var fenzi = 0.0
          var u_fenmu = 0.0
          var v_fenmu = 0.0
          for (i <- 0 until length_of_rating by 1) {
            fenzi += (u_rating(i) - u_aver) * (v_rating(i) - v_aver)
            u_fenmu += pow(u_rating(i) - u_aver, 2)
            v_fenmu += pow(v_rating(i) - v_aver, 2)
          }
          val fenmu = pow(u_fenmu * v_fenmu, 0.5)
          if (fenmu != 0) {
            similarity = fenzi / fenmu
          }
        }
        (x._1,similarity)
      }).filter(_._2>0)

//      val item_similarity = item_pairs.flatMap(x=>List(Tuple2(x._1._1,List(Tuple2(x._1._2,x._2))), Tuple2(x._1._2,List(Tuple2(x._1._1,x._2))))).reduceByKey((x, y)=>List.concat(x,y)).collect()
//      val item_sim = scala.collection.mutable.HashMap.empty[Int,List[Tuple2[Int,Double]]]
//      for (u_s<-item_similarity){
//        item_sim(u_s._1) = u_s._2
//      }
//
//      val user_business= train_ratings.map(x=>((x._1,x._2),x._3)).collect()
//      val user_business_rating = scala.collection.mutable.HashMap.empty[Tuple2[Int,Int],Float]
//      for (u_b<- user_business){
//        user_business_rating(u_b._1) = u_b._2
//      }
//
//      val user_average = train_ratings.map(x=> (x._1, List(x._3))).reduceByKey((x,y)=> List.concat(x,y)).map(
//        x=>(x._1, x._2.sum / x._2.length)).collect()
//      val user_aver = scala.collection.mutable.HashMap.empty[Int,Float]
//      for (u_a <- user_average){
//        user_aver(u_a._1) = u_a._2
//      }


      val item_similarity = item_pairs.collect()
      val item_sim = scala.collection.mutable.HashMap.empty[Tuple2[Int,Int],Double]
      for (i_s <- item_similarity){
        item_sim(i_s._1) = i_s._2
      }

      val all_other_rated_item = scala.collection.mutable.HashMap.empty[Int,List[Tuple2[Int,Float]]]
      for (u_i <-user_business_in_train.collect()){
        all_other_rated_item(u_i._1) = u_i._2
      }

      val result = test_ratings.map(x=>{
        val user = x._1
        val active_item = x._2
        var predict_rating = 3.5
        var fenzi = 0.0
        var fenmu = 0.0
        var other_rated_items = List.empty[(Int,Float)]
        if (all_other_rated_item.contains(user)){
          other_rated_items = all_other_rated_item(user)
        }
        if (other_rated_items.nonEmpty) {
          val m = 1/other_rated_items.length.toDouble
          for (compare_item <- other_rated_items){
            val item_pair = Tuple2(List(active_item,compare_item._1).min,List(active_item,compare_item._1).max)
            var similarity = m
            if (item_sim.contains(item_pair)){
              similarity = item_sim(item_pair)
            }
            fenzi +=compare_item._2*similarity
            fenmu +=abs(similarity)
          }
          if (fenmu!=0){
            predict_rating =fenzi/fenmu
          }
        }
        ((user, active_item), predict_rating,x._3)
      })

//      val result = test_ratings.map(x=> {
//        val user = x._1
//        val active_item = x._2
//        var predict_rate = 3.5
//        if (user_aver.contains(user)){
//          predict_rate = user_aver(user).toDouble
//        }
//        val item_similarity = item_sim.getOrElse(active_item,List.empty).sortBy(x=>x._2)
//        //        println(user_similarity)
//        if (item_similarity.nonEmpty){
//          var compare_item_count = 0
//          var fenzi = 0.0
//          var fenmu = 0.0
//          for (i_s<-item_similarity) {
//            if (compare_item_count <= 10) {
//              val compare_item = i_s._1
//              var similarity = i_s._2
//              if (user_business_rating.contains(Tuple2(user, compare_item))){
//                val rui = user_business_rating(Tuple2(user, compare_item))
//                compare_item_count += 1
//                fenzi += rui * similarity
//                fenmu += abs(similarity)
//              }
//            }
//          }
//          if (fenmu != 0){
//            //            println(fenzi/fenmu)
//            predict_rate = fenzi/fenmu
//          }
//        }
//        ((user,active_item),predict_rate,x._3)
//      })


//      val RMSE = pow(result.map(x=>pow(x._2 - x._3,2)).mean(),0.5)
//      print(RMSE)

      val pw = new PrintWriter(outputfile)
      pw.println("user_id, business_id, prediction")
      for (pair <- result.collect())
        pw.println(uids(pair._1._1)+","+bids(pair._1._2)+","+pair._2.toString)
      pw.close()
//      val end = System.currentTimeMillis()
//      println(end-start)
    }

  }
}
