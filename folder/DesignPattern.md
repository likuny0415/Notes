# Design Pattern

## Structual Design Pattern

Structural design pattern is a ease to design by providing a simple way to realize relationships among entities.



### Adapter

Image you have a stock market app which use XMl format. Now you want to import a 3D library which use JSON format, you need to revise your code to make compatiable with 3D library, which may breaking your existing code...

Now!

You can use a adapter to convert the interface of one object so that another object can understand it. In this case, adapter helps to transfer XML format into JSON format which can help you collborate with 3D library, this is will not break the initial interface and code!

##### Applicability

- When you want to use some existing code, but the interface isn't compatiable with the new class
- When there are mutliple layers, use adapter as a translator

##### Reference

[Adapter](https://refactoring.guru/design-patterns/adapter)

[Java Version](https://refactoring.guru/design-patterns/adapter/java/example)



### Bridge



### Decorator

Image you want to create a  `notifier` , which send important notification to users.

```
Notifier
	send
```

Client can use it to instantiate the desired notifier class and use for further notifications

Now.. you want to send not only by email, but also by GMS, Facebook, Slack, WeChat.. You can just inherit the notifier class

```
				Notifier
	GMS  Facebook  Slack
```

However, you want to notify more than one platform at once... Things will be complicated because of inheritance has some caveats: 1. Inheritance is static, once you assigned it, you can change it (Student can be changed to Employee) 2. Subclass could only have one parent at the same time

```
					Notifier
		GMS  Facebook  Slack
G+SMS  G+F  		F+S	    S+G
```

So, we could use **Aggregation/Composition** to help finish the job, which delegate the job to another object. Now you can change the type at runtime. **Aggregation/Composition** is the key behind Decorator

##### Real Wolrd Analogy

Wearing clothes: If you cold, you can wear more clothes, if it is raining, you can wear a raincoat..You can add or remove any cloth whenever you need it..

##### Reference

[Decorator](https://refactoring.guru/design-patterns/decorator)

[Decorator Example in Java](https://refactoring.guru/design-patterns/decorator/java/example)

