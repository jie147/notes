#**解析spark graphx shortest path**

  在GraphX中shortest path使用的是Pregel接口实现的。

  源代码：
```scala
/**
 * Computes shortest paths to the given set of landmark vertices, returning a graph where each
 * vertex attribute is a map containing the shortest-path distance to each reachable landmark.
 */
object ShortestPaths {
  /** Stores a map from the vertex id of a landmark to the distance to that landmark. */
  type SPMap = Map[VertexId, Int]

  private def makeMap(x: (VertexId, Int)*) = Map(x: _*)

  private def incrementMap(spmap: SPMap): SPMap = spmap.map { case (v, d) => v -> (d + 1) }

  private def addMaps(spmap1: SPMap, spmap2: SPMap): SPMap =
    (spmap1.keySet ++ spmap2.keySet).map {
      k => k -> math.min(spmap1.getOrElse(k, Int.MaxValue), spmap2.getOrElse(k, Int.MaxValue))
    }.toMap

  /**
   * Computes shortest paths to the given set of landmark vertices.
   *
   * @tparam ED the edge attribute type (not used in the computation)
   *
   * @param graph the graph for which to compute the shortest paths
   * @param landmarks the list of landmark vertex ids. Shortest paths will be computed to each
   * landmark.
   *
   * @return a graph where each vertex attribute is a map containing the shortest-path distance to
   * each reachable landmark vertex.
   */
  def run[VD, ED: ClassTag](graph: Graph[VD, ED], landmarks: Seq[VertexId]): Graph[SPMap, ED] = {
    val spGraph = graph.mapVertices { (vid, attr) =>
      if (landmarks.contains(vid)) makeMap(vid -> 0) else makeMap()
    }

    val initialMessage = makeMap()

    def vertexProgram(id: VertexId, attr: SPMap, msg: SPMap): SPMap = {
      addMaps(attr, msg)
    }

    def sendMessage(edge: EdgeTriplet[SPMap, _]): Iterator[(VertexId, SPMap)] = {
      // 将dst节点的attr的value增1
      val newAttr = incrementMap(edge.dstAttr)
      //
      if (edge.srcAttr != addMaps(newAttr, edge.srcAttr)) Iterator((edge.srcId, newAttr))
      else Iterator.empty
    }

    Pregel(spGraph, initialMessage)(vertexProgram, sendMessage, addMaps)
  }
}
```
  
  1. 初始图 spGraph：Graph[ Map(), ED] ，vertexId 在 landmarks 中时Map中的值为（VertexId，0），否则Map中为 null 。
  2. 初始化图：使用 vertexProgram 函数和 initialMessage 对 spGraph 进行初始化。将会得到一个初始化图 g:GraphX(Map(),ED)
  3. 使用sendMessage和消息合并函数addMaps，将会得到第一次消息交互信息，messages：VertexRDD，其中attr是一个Map。
  4. 接下来就进入while循环了，首先，将messages和图g使用顶点程序合并。
  5. 又是一个sendMessage和消息合并，得到一个message。当条件满足事调转到4，否则跳出循环，在此程序中，跳出循环的条件是当图中无消息交互，即messages的大小为0时跳出循环。
  
##测试
  测试代码：
```scala
val shortestPaths = Set(
        (1, Map(1 -> 0, 4 -> 2)), (2, Map(1 -> 1, 4 -> 2)), (3, Map(1 -> 2, 4 -> 1)),
        (4, Map(1 -> 2, 4 -> 0)), (5, Map(1 -> 1, 4 -> 1)), (6, Map(1 -> 3, 4 -> 1)))
      val edgeSeq = Seq((1, 2), (1, 5), (2, 3), (2, 5), (3, 4), (4, 5), (4, 6)).flatMap {
        case e => Seq(e, e.swap)
      }
      val edges = sc.parallelize(edgeSeq).map { case (v1, v2) => (v1.toLong, v2.toLong) }
      val graph = Graph.fromEdgeTuples(edges, 1)
      val landmarks = Seq(1, 4).map(_.toLong)
      val results = ShortestPaths.run(graph, landmarks).vertices.collect.map {
        case (v, spMap) => (v, spMap.mapValues(i => i))
      }
      
```
  测试数据如下：
  ![shortestPaths](./image/shortestPaths.png)
  landmarks 的数据点为 [ 1, 4 ]时 其结果如下：（缺了个3和5节点的数据）
```
Array((1,Map(1 -> 0, 4 -> 2)), 
      (2,Map(1 -> 1, 4 -> 2)), 
      (4,Map(4 -> 0, 1 -> 2)), 
      (6,Map(4 -> 1, 1 -> 3))
)
```
  landmarks 的数据为  [ 1, 6 ] 时，其结果如下：
```
Array(
(1,Map(1 -> 0, 6 -> 3)), 
(2,Map(1 -> 1, 6 -> 3)), 
(3,Map(6 -> 2, 1 -> 2)), 
(4,Map(6 -> 1, 1 -> 2)), 
(5,Map(1 -> 1, 6 -> 2)), 
(6,Map(6 -> 0, 1 -> 3))
)
```