# Web

### DTO(Data Transfer Object)

DTO: An object that used to encapsulate data, send it from one subsystem of an application to another.

### REST API

##### What is REST API?

API is set of rules define how application interact and communicate with each other. REST is a set of standard architectural constraints, it is not a protocol or a standard.

##### REST API design principles:

Basic: The application or service doing the accessing is called the client, and the application or service containing the resource is called the server.

##### Architectural Constraints:

1. Uniform interface
   - All API requests for same resources should be identical
2. Client-Server decoupling
   - Client and Server application should be independent
3. Statelessness
   - No client information is stored between GET requests, each request is separate and unconnected
4. Cacheability
   - Cache everytime possible, performance boost on server side and scalability on server side
5. Layer system architecture
   - Don't assume client directly contact with server, they contact pass multiple intermediary
6. Code on demand(Optionl)
   - Most time send static resources, but sometimes send executable code

##### How REST APIs Work

REST API communicate via http request to perform standard CRUD functions. GET retrieve a record, POST create a record, PUT update a record and DELETE remove a record.

##### Important

Headers and parameters are important in the HTTP methods of RESTful API HTTP request, they contain important **identifier** information as to request's metadata, authorization, URI, caching, cookies, and more. Request header and response header each have their own HTTP connection information and status code. 

### Promise

##### What is a promise?

A promise is an **object** represent the eventual completion or failure of an asynchronous operation. It is a returned object to which you attach **callbacks**, instead of pass callbacks into a function.

```javascript
createAudioFileAsync(audioSettings).then(successCallback, failureCallback);
```

 ##### Advantages

1. Callbacks will never invoke before the completion of the current run of the JS Event loop becuase of `then()`
2.  Callbacks in callbacks will also be invoked(callbacks in successCallback will also be invoked)
3. Many callbacks will be invoked by calling `then()` serveral times. They will be invoked inorder
4. **`Chaining`** 

##### Chaining

```javascript
const promise2 = doSomething().then(successCallback, failureCallback);
```

Any callbacks added to promise2 get **queued** behind the promise after success or failure callback.





### Async, Await

##### Why asynochronous

If you run a program which depend on another program to run at first, if the second program is not loading properly, you will never be able to use the first program, which is bad and slow. This is because Javascript is **singled-threaded** which means **everything is blocked** until an operation completes

##### What is thread 

A thread is basically a process that a program can use to complete tasks. Each thread can only do a single task at once. `Task A --> Task B --> Task C `

Javascript use `workers` to run expensive processes off the main thread so that user interaction is no longer blocked.

##### Async

An async function is a function that knows what to expect after `await` keyword being used to invoke asynochronous code.

```javascript
async function hello() {return "Hello"}
hello();
// This is will return a Promise object

// Arrow
let hello = async () => "Hello";

// hello().then(console.log) is same with
hello().then((value) => console.log(value))
// return type =>
//[[Prototype]]: Promise
//[[PromiseState]]: "fulfilled"
//[[PromiseResult]]: undefined
```

So, the `await` keyword is added to functions to tell them to return a `Promise` instead just return the value

##### Await

You can put `await` in front of any async promise-based function to pause your code on that line until the promise fulfills, then return the resulting value.

##### Rewrite with async and await

Before

```javascript
fetch(".jpb")
  .then(response => {
  if (!response.ok){
    throw new Error(`Error code is ${response.code}`);
  }
  return response.blob() // callback
})
  .then(myBlob => {
  // some variables and action
})
  .catch(e => {
  console.log(e);
})
```

After

```javascript
async function myFetch() {
  let response = await fetch(".jpg");
  if (!response.ok){
    // do something
  }
  // some variables and action
}

myFetch()
  .catch(console.log)
```

Difference

- No more `then()` blocks

##### 