
#**Spark GraphX LabelPropagation**
LabelPropagation 使用的是LPA算法，LPA算法描述如下：(参考：http://blog.csdn.net/cleverlzc/article/details/39494957)
> 标签传播算法（LPA）的做法比较简单：
> 第一步: 为所有节点指定一个唯一的标签；
> 第二步: 逐轮刷新所有节点的标签，直到达到收敛要求为止。对于每一轮刷新，节点标签刷新的规则如下:
  >> 对于某一个节点，考察其所有邻居节点的标签，并进行统计，将出现个数最多的那个标签赋给当前节点。当个数最多的标签不唯一时，随机选一个。

##LabelPropagation run方法接口
```scala
def run[VD, ED](
	graph: Graph[VD, ED], 
	maxSteps: Int
): Graph[VertexId, ED]
```
  参数解析：
  `garph`： the graph for which to compute the community affiliation
  `maxSteps`： the number of supersteps of LPA to be performed. Because this is a static implementation, the algorithm will run for exactly this many supersteps.

  测试数据：
  ![测试数据图](./images/jietest.png)
  
  测试代码：
```scala
// Construct a graph with two cliques connected by a single edge
val n = 5
val clique1 = for (u <- 0L until n; v <- 0L until n) yield Edge(u, v, 1)
val clique2 = for (u <- 0L to n; v <- 0L to n) yield Edge(u + n, v + n, 1)
val twoCliques = sc.parallelize(clique1 ++ clique2 :+ Edge(0L, n, 1))
val graph = Graph.fromEdges(twoCliques, 1)
// Run label propagation
val labels = LabelPropagation.run(graph, n * 4).cache()
labels.vertices.collect.foreach(println)
```
  测试结果如下：
```
(4,0)
(0,0)
(6,5)
(8,5)
(10,5)
(2,0)
(1,0)
(3,0)
(7,5)
(9,5)
(5,5)
```


##源码简单分析
  LabelPropagation 的 run 方法
```scala
def run[VD, ED: ClassTag](graph: Graph[VD, ED], maxSteps: Int): Graph[VertexId, ED] = {
    require(maxSteps > 0, s"Maximum of steps must be greater than 0, but got ${maxSteps}")

    val lpaGraph = graph.mapVertices { case (vid, _) => vid }
    def sendMessage(e: EdgeTriplet[VertexId, ED]): Iterator[(VertexId, Map[VertexId, Long])] = {
      Iterator((e.srcId, Map(e.dstAttr -> 1L)), (e.dstId, Map(e.srcAttr -> 1L)))
    }
    def mergeMessage(count1: Map[VertexId, Long], count2: Map[VertexId, Long])
      : Map[VertexId, Long] = {
      (count1.keySet ++ count2.keySet).map { i =>
        val count1Val = count1.getOrElse(i, 0L)
        val count2Val = count2.getOrElse(i, 0L)
        i -> (count1Val + count2Val)
      }.toMap
    }
    def vertexProgram(vid: VertexId, attr: Long, message: Map[VertexId, Long]): VertexId = {
      if (message.isEmpty) attr else message.maxBy(_._2)._1
    }
    val initialMessage = Map[VertexId, Long]()
    Pregel(lpaGraph, initialMessage, maxIterations = maxSteps)(
      vprog = vertexProgram,
      sendMsg = sendMessage,
      mergeMsg = mergeMessage)
  }
```
  prepel 会将图初始化为 Graph[ VertexId, ED ] ，其中 VertexId 就是自身 Id 。
  随后将会进入迭代阶段，迭代过程如下：
```flow
st=>start: 开始
e=>end: 结束
initGraph=>operation: 初始化迭代图
sendMes=>operation: 发送消息
meMes=>operation: 融合消息
vertexPro=>operation: 节点程序
condition=>condition: 是否满足条件?

st->initGraph->condition->vertexPro->sendMes->meMes->condition
condition(yes)->vertexPro
condition(no)->e
```
  
  发送消息：（Iterator发给谁了？？Iterator[(VertexId, A) 将A发给了vertexId 。）triplet两端消息的交互，发送Map（VertexId -> 1L）
  消息融合：将接受到的value进行融合。
  节点程序：返回的是最大值的ID号。

  整体思路：（需对prepel接口有所了解。）
  > 1. 在进入while循环之前，对图进行初始化得到 g：Graph[VertexId, ED]，其中VerTexId为空，随后将会有 mapReduceTriplets ，每个节点将会得到多个节点发来信息，信息是一个Map但是其中没有内容，通过消息融合后将会得到一个 messages：RDD[ VertexId, Map(( VertexId, 0 ) * ) ] 。
  > 2. 在while循环内（迭代次数到达到或无消息进行处理时则跳出循环）
  > + 第一步，筛选出周边节点对那个节点的支持力度大，并将其节点Id 放到节点attr中（如果有多个最大值将随机挑选一个），返回图 g ： Graph[VertexId, ED] 。（第一次进入循环时，messages中value的值为0 ，使用maxBy函数将会随机挑选一个节点Id。）
  > + 第二步，消息的发送与合并。发送：将此节点支持的节点Id发送出去。合并：统计周边邻节点对哪些节点的支持率如何。将统计到数据汇集到一个Map中（此Map有点类似Set中放着Tuple） messages：RDD[ VertexId, Map(( VertexId, 0 ) * ) ] 。