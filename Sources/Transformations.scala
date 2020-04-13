// READING DATA

val alldata0 = spark.read.parquet("hdfs:///data/alldata_index.parquet")
val pheno = spark.read.parquet("hdfs:///data/pheno.parquet")


// EXPLODE FUNCTION

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{array, col, explode, lit, struct}


def toLong(df2: DataFrame, by: Seq[String]): DataFrame = {
  val (cols, types) = df2.dtypes.filter{ case (c, _) => !by.contains(c)}.unzip
  require(types.distinct.size == 1, s"${types.distinct.toString}.length != 1")      
  val kvs = explode(array(
    cols.map(c => struct(lit(c).alias("key"), col(c).alias("value"))): _*
  ))
  val byExprs = by.map(col(_))
  df2
    .select(byExprs :+ kvs.alias("_kvs"): _*)
    .select(byExprs ++ Seq($"_kvs.key", $"_kvs.value"): _*)
}


// REQUIRED CLASSES

import org.apache.spark.ml.feature.LabeledPoint 

case class Pt(IID: String, features: Array[(Int, Double)], Obesity: Double)

case class pSNP(IID: String, Obesity: Double, index: Int, value: Double)

case class PtGroup(IID: String, Index: Array[Int], Value: Array[Double], Obesity: Double)

case class LPP(IID:String,LP:LabeledPoint)


// REQUIRED FUNCTIONS

import org.apache.spark.sql._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.feature.LabeledPoint 



def Rows2SNP(data:DataFrame): Dataset[pSNP] = data.map((f) => pSNP(IID = f(0).asInstanceOf[String], Obesity = f(1).asInstanceOf[Double], index = f(2).asInstanceOf[Int], value = f(3).asInstanceOf[Double]))



def snp2pt(snpDs:Dataset[pSNP]): Dataset[Pt] = {
  snpDs
    .map((f) => (
      Pt(
        IID=f.IID,
        Obesity=f.Obesity,
        features = Array((f.index, f.value))
      )))
}


def groupByPatient(pDs: Dataset[Pt]):Dataset[Pt] ={
    pDs
        .groupByKey(_.IID)
        .reduceGroups((f1,f2) =>
          Pt(IID=f1.IID,
            features=f1.features ++ f2.features,
            Obesity=f1.Obesity
          )
        )
        .map(_._2)
        .map((f)=>
         Pt(IID=f.IID,
            features=f.features.sortBy(_._1),
            Obesity=f.Obesity
           )
        )
}



def grouped2LabeledPoint(pDs: Dataset[Pt]):Dataset[LPP] = {
    pDs
    .map((f) => PtGroup(IID=f.IID,Index=f.features.unzip._1,Value=f.features.unzip._2,Obesity=f.Obesity))
    .map((f) => LPP(IID=f.IID,LP=LabeledPoint(f.Obesity,Vectors.sparse(19581634,f.Index,f.Value))))
}

// LOOP FUNCTION

spark.conf.set("spark.sql.shuffle.partitions", 2100)

def looplabeledpoint(selectColumns:Seq[Seq[String]]): Unit={
    
    for (colname <- selectColumns){ 

        //selecting 2 columns per iteration
        val alldata = alldata0.select((colname :+ "index").map(col): _*)

        //Long format
        val prova = toLong(alldata.select(alldata.columns.map(c => upper(col(c)).alias(c)): _*), Seq("index"))

        
        val databypt = prova.withColumn("index",col("index").cast("int")).withColumn("value",col("value").cast("double"))
    
        //Merging Long format + phenotype
        val data = databypt
                .join(broadcast(pheno), $"IID" === $"key", "left")
                .select("IID","Obesity","index","value")
    
        //Grouping data + final labeled point
        val labeledpoint = data.transform(Rows2SNP)
                        .transform(snp2pt)
                        .transform(groupByPatient)
                        .transform(grouped2LabeledPoint)  
                        
        //writing labeledpoint file
        labeledpoint.write.option("compression","snappy").mode("append").partitionBy("IID").parquet("hdfs:///data/labeledpoint.parquet")
        
        //writing completeds file
        colname.toDF.write.mode("append").parquet("hdfs:///data/completeds.parquet")
    }
    
}

// LOOP

val allcolumns = alldata0.columns.toSeq.filter(_ != "index_0").filter(_ != "SNP").slice(0,4988)


val conf = sc.hadoopConfiguration
val fs = org.apache.hadoop.fs.FileSystem.get(conf)
val exists = fs.exists(new org.apache.hadoop.fs.Path("hdfs:///data/completeds.parquet"))


val n = 2


if (exists){
    val calculats: Set[String] = spark.read.parquet("hdfs:///data/completeds.parquet").collect.map(_(0).toString).toSet
    val remaining = allcolumns.filterNot(calculats)
    looplabeledpoint(remaining.sliding(n,n).toSeq)
} else{
    val remaining = allcolumns
    looplabeledpoint(remaining.sliding(n,n).toSeq)
}
