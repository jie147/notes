@autor:[jie](https://github.com/jie147)
  个人笔记：（不对之处请指正）
  从 DataFrame 开始捋吧，DataFrame的数据类型：
```
type DataFrame = Dataset[Row]  
```
  可见，DataFrame 就是一个 Dataset 数据集，但是 Dataset 是一个泛型。
```
class Dataset[T] extends Serializable
```
  这样就能很好的理解 DataFrame 其实就是 Dataset 数据集，其中包含的数据类型就是 Row . 如果理解了Row 就基本理解 DataFrame 了。现在就差Row 数据类型了。那Row类型是啥样的呢？
```
import org.apache.spark.sql._

val row = Row(1, true, "a string", null)
// row: Row = [1,true,a string,null]
val firstValue = row(0)
// firstValue: Any = 1
val fourthValue = row(3)
// fourthValue: Any = null
```
  从中可以感觉到Row在某种意义上就犹如 scala 中的 Array 一样能够包罗万向。下面是一个获取 Row 中值的一个例子。`注意最后一行，当 Boolean 类型为 true 时，为空。`
```scala
// using the row from the previous example.
val firstValue = row.getInt(0)
// firstValue: Int = 1
val isNull = row.isNullAt(3)
// isNull: Boolean = true
```

  [Dataset](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.Dataset) 类的操作函数还是有相当多的，想要用好 DataFrame 还是得多看Dataset的[API](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.Dataset)的。