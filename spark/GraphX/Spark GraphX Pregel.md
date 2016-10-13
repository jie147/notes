#**解析Spark GraphX Pregel**
 后续会有所添加
  [官网 pregel API详解](http://spark.apache.org/docs/latest/graphx-programming-guide.html#pregel-api)
##GraphX 中 Pregel 目的
###pregel起源
  pregel是googel公司的大型图计算平台。[这篇文章](http://www.360doc.com/content/11/0609/19/6986090_122743127.shtml)对prepel有详细的介绍。
###在GraphX中的Pregel
  在GraphX中Pregel接口主要实现的是图计算的遍历和迭代问题，图的迭代算法是非常常见的算法，因此Pregel接口的作用就不言而喻了。
##Pregel如何实现迭代
  pregel虽然是构建在spark GraphX上的高级接口，但是pregel接口并未因此而失去灵活性。
###源码
```scala
def apply[VD: ClassTag, ED: ClassTag, A: ClassTag]
     (graph: Graph[VD, ED],
      initialMsg: A,
      maxIterations: Int = Int.MaxValue,
      activeDirection: EdgeDirection = EdgeDirection.Either)
     (vprog: (VertexId, VD, A) => VD,
      sendMsg: EdgeTriplet[VD, ED] => Iterator[(VertexId, A)],
      mergeMsg: (A, A) => A)
    : Graph[VD, ED] =
  {
    require(maxIterations > 0, s"Maximum number of iterations must be greater than 0," +
      s" but got ${maxIterations}")

    var g = graph.mapVertices((vid, vdata) => vprog(vid, vdata, initialMsg)).cache()
    // compute the messages
    var messages = GraphXUtils.mapReduceTriplets(g, sendMsg, mergeMsg)
    var activeMessages = messages.count()
    // Loop
    var prevG: Graph[VD, ED] = null
    var i = 0
    while (activeMessages > 0 && i < maxIterations) {
      // Receive the messages and update the vertices.
      prevG = g
      g = g.joinVertices(messages)(vprog).cache()

      val oldMessages = messages
      // Send new messages, skipping edges where neither side received a message. We must cache
      // messages so it can be materialized on the next line, allowing us to uncache the previous
      // iteration.
      messages = GraphXUtils.mapReduceTriplets(
        g, sendMsg, mergeMsg, Some((oldMessages, activeDirection))).cache()
      // The call to count() materializes `messages` and the vertices of `g`. This hides oldMessages
      // (depended on by the vertices of g) and the vertices of prevG (depended on by oldMessages
      // and the vertices of g).
      activeMessages = messages.count()

      logInfo("Pregel finished iteration " + i)

      // Unpersist the RDDs hidden by newly-materialized RDDs
      oldMessages.unpersist(blocking = false)
      prevG.unpersistVertices(blocking = false)
      prevG.edges.unpersist(blocking = false)
      // count the iteration
      i += 1
    }
    messages.unpersist(blocking = false)
    g
  } // end of apply
```
  可以看出Pregel接口的实现就20几行的代码。那么它是如何工作的呢？？
###源码分析
  prepel 接口的流程图：
```flow
start=>start: 开始
end=>end: 结束
initOp=>operation: 初始化图
outWhileMes=>operation: 消息发送与合并
condition=>condition: 小于maxIterations & 消息交互量>0
vertexPro=>operation: 节点程序
Mes=>operation: 消息发送与合并

start->initOp->outWhileMes->condition
condition(yes)->vertexPro->Mes->condition
condition(no)->end
```
  
  从传入函数的图开始讲起，传入的图graph: Graph[VD, ED]，这是一个泛型接口，增加了一定灵活性。
  下面主要解析Pregel中的操作：
  1. 初始化图：使用mapVertices 将初值 initialMsg 使用 `顶点程序` 合并到图中。initialMsg与图迭代产生的messages的数据类型相同。（数据类型相同的目的是啥？）
  2. 消息的首次发送与合并，这个是必须要做的，目的是测试消息的交互数量，如果没有消息交互就直接结束了。有消息的交互才需要迭代。
  3. 1-2步都是在迭代外围，从现在开始就要进行迭代了。首先是节点程序，由于在第2步中产生了messages，但是并未合并到图中，所以需要合并到图中。
  4. 第4步又是消息的发送与合并，产生messages，在迭代范围内会跳到第3步，并将messages合并到图中。
  
##实例s
###pagerank
###LabelPropagation
###ShortestPaths
##问题s
### 1. 消息发送函数，消息是向哪里发？
```
sendMsg: EdgeTriplet[VD, ED] => Iterator[(VertexId, A)]
```
  上面的函数是prepel接口sendMsg函数，其中返回的结果是Iterator[(VertexId, A) 。那么问题来了，这个Iterator会向那个节点发送呢？
  查看`GraphXUtils.mapReduceTriplets` 源码：
```scala
private[graphx] def mapReduceTriplets[VD: ClassTag, ED: ClassTag, A: ClassTag](
      g: Graph[VD, ED],
      mapFunc: EdgeTriplet[VD, ED] => Iterator[(VertexId, A)],
      reduceFunc: (A, A) => A,
      activeSetOpt: Option[(VertexRDD[_], EdgeDirection)] = None): VertexRDD[A] = {
def sendMsg(ctx: EdgeContext[VD, ED, A]) {
      mapFunc(ctx.toEdgeTriplet).foreach { kv =>
        val id = kv._1
        val msg = kv._2
        if (id == ctx.srcId) {
          ctx.sendToSrc(msg)
        } else {
          assert(id == ctx.dstId)
          ctx.sendToDst(msg)
        }
      }
    }
    g.aggregateMessagesWithActiveSet(sendMsg, reduceFunc, TripletFields.All, activeSetOpt)
```
  可以看出mapReduceTriplets函数现在也是在调用aggregateMessagesWithActiveSet函数，可以看出sendMsg函数调用的是mapFunc函数（这个也就是自己实现的sendMsg函数），其返回的结果是Iterator[(VertexId, A)] ，在foreach函数中有 if 函数，当Iterator中的VertexId是triplet的源节点时，将A发送给源节点。
  **总结：sendMsg 函数的返回值 `Iterator[(VertexId, A)]` ，其中VertexId 指明了需要发送给那个节点。**
### 2. activeMessages 的疑问？
  activeMessages 代表了消息的交互数量，当为0时，将会结束迭代。
### 3. 为啥需要initialMsg？其目的是啥？
  