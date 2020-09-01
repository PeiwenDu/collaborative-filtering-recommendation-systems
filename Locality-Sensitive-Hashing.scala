import org.apache.spark.{SparkConf, SparkContext}
import java.io.PrintWriter

object peiwen_du_task1 {
  def main(args: Array[String]): Unit = {
    val inputfile = args(0)
    val outputfile = args(1)

//    val inputfile = "/Users/peiwendu/Downloads/public_data/yelp_train.csv"
//    val outputfile = "peiwen_du_task1"

    val conf = new SparkConf()
    conf.setAppName("hw3.1")
    conf.setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

//    val start = System.currentTimeMillis()
    val rdd = sc.textFile(inputfile)
    val header = rdd.first()
    val test_data = rdd.filter(x => x != header).map(x => x.split(","))

    val users = test_data.map(_(0)).distinct().collect()
    val businesses = test_data.map(_(1)).distinct().collect()
    val user_index = scala.collection.mutable.HashMap.empty[String, Int]
    val business_index = scala.collection.mutable.HashMap.empty[String, Int]
    var index_of_users = 0
    var index_of_business = 0
    for (e <- users) {
      user_index(e) = index_of_users
      index_of_users += 1
    }
    for (e<- businesses){
      business_index(e) = index_of_business
      index_of_business+=1
    }

    val character_matrix = test_data.map(x=>(business_index(x(1)),user_index(x(0)))).groupByKey().mapValues(_.toList)
    val characters = character_matrix.collect()

    val b_c = scala.collection.mutable.HashMap.empty[Int, List[Int]]
    for (b<-characters){
      b_c(b._1) = b._2
//      println(b._2)
    }

    val m = users.length
//    println(m)

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
//    for (i<-signatiure.take(2)){
//      println(i._1)
//      println(i._2)
//    }

//
//
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
//      println(whole)
      for (i <- 0 until length) {
        for (j <- i+1 until length) {
          res += Tuple2(Tuple2(whole(i), whole(j)), 1)
        }
      }
      res.toList
    }).reduceByKey((x, y) => x).map(x => x._1)

//        for (i<-candidate_pair.take(2)){
//          println(i._1)
//          println(i._2)
//        }

    val similar_pair = candidate_pair.map(x=> {
      val bid1 = x._1
      val bid2 = x._2
      val character1 = b_c(bid1).toSet
//      println(character1)
      val character2 = b_c(bid2).toSet
//      println(character2)
      val similarity = character1.intersect(character2).size.toFloat / character1.union(character2).size
//      println(similarity)
      val pair = List(businesses(bid1),businesses(bid2)).sorted
//      println(pair)
      val first = pair(0)
      val second = pair(1)
      Tuple3(first, second, similarity)
    }).filter(_._3>=0.5).sortBy(x => x._1)

//    for (i<-similar_pair.take(2)){
//      println(i._1)
//      println(i._2)
//      println(i._3)
//    }
    val pw = new PrintWriter(outputfile)
    pw.println("business_id_1, business_id_2, similarity")
    for (pair<- similar_pair.collect())
      pw.println(pair._1+","+pair._2+","+pair._3.toString)
    pw.close()


//    val end = System.currentTimeMillis()
//    println(end-start)
//
//    val similar_pairs = similar_pair.map(x=>(x._1,x._2)).collect()
//    val vali_data = sc.textFile("/Users/peiwendu/Downloads/public_data/pure_jaccard_similarity.csv")
//    val head = vali_data.first()
//    val vali_pair = vali_data.filter(x=>x != header).map(x=>x.split(",")).map(x=>(x(0),x(1))).collect()
//    val ture_positive = similar_pairs.toSet.intersect(vali_pair.toSet).size
//    val false_positive = similar_pairs.toSet.diff(vali_pair.toSet).size
//    println(false_positive)
//    val false_negaive = vali_pair.toSet.diff(similar_pairs.toSet).size
//    println(false_negaive)
//    val precision = ture_positive/(ture_positive+false_positive).toFloat
//    val recall = ture_positive/(ture_positive+false_negaive).toFloat
//
//    println(precision)
//    println(recall)

  }
}
