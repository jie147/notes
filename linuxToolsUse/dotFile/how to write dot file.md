#无向图

```
graph graphname {
	a -- b -- c;
	b -- d;
}
```
  参考图片：
![无向图](https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/DotLanguageUndirected.svg/168px-DotLanguageUndirected.svg.png)
#有向图
```
digraph graphname {
     a -> b -> c;
     b -> d -> e -> f;
 }
```
  参考图片：
![有向图](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/DotLanguageDirected.svg/220px-DotLanguageDirected.svg.png)

#属性
DOT语言中，可以对节点和边添加不同的属性。这些属性可以控制节点和边的显示样式，例如颜色，形状和线形。可以在语句和句尾的分号间放置一对方括号，并在其中中放置一个或多个属性-值对。多个属性可以被逗号和空格（, ）分开。节点的属性被放置在只包含节点名称的表达式后。
```
graph graphname {
     // label属性可以改变节点的显示名称
     a [label="Foo"];
     // 节点形状被改变了
     b [shape=box];
     // a-b边和b-c边有相同的属性
     a -- b -- c [color=blue];
     b -- d [style=dotted];
 }
```
  参考图片：
  ![图的属性](https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/DotLanguageAttributes.svg/168px-DotLanguageAttributes.svg.png)
 
 例子：
```
digraph g {
	node [shape=plaintext];
	A1 -> B1;
	A2 -> B2;
	A3 -> B3;
	
	A1 -> A2 [label=f];
	A2 -> A3 [label=g];
	B2 -> B3 [label="g'"];
	B1 -> B3 [label="(g o f)'" tailport=s headport=s];

	{ rank=same; A1 A2 A3 }
	{ rank=same; B1 B2 B3 } 
}
```
#参考：
https://en.wikipedia.org/wiki/DOT_(graph_description_language)
http://www.ibm.com/developerworks/cn/aix/library/au-aix-graphviz/index.html