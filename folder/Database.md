# DataBase

### UML Diagram Tutorial

##### Reference

[Tutorial](https://www.visual-paradigm.com/guide/uml-unified-modeling-language/uml-class-diagram-tutorial/)

### Aggregation

Imaging you have a computer...

```
Computer:
	IP
	Name
	CPU
	GPU1Size
	GPU1Spec
	GPU2Size
	GPU2Spec
	...
```

As you attach more and more components to the computer, it will be more complicated to manange it. So, the solution is to separate the computer from its components.

```
Computer:
	IP
	Name
	component1:
	component2:
	...

component1:
	Size
	Capacity
	...
	
component2:
	Size
	Capacity
	Latency
	...
```

As a result, in database, we could build two table to link them, main table keep track of computer information and another keep track of its components. 

**Important**

- Open diamond point to the parent (**aggregated**) class, which implies 1...* relationship
- Since the component can live by itself, it **PK** should be its {type, size, capacity..}, computerName should be its **FK**

##### Reference

[Aggregation](https://web.csulb.edu/colleges/coe/cecs/dbdesign/dbdesign.php?page=aggregate.php)

-----

### Composition

Same concept with Aggregation..

**Important**

- Use filled-in Diamond instead of an open one
- Computer components **can't live by its own**, there are created because of computer and will be deleted if the computer is destroyed. 

