# JavaScript Syntax

### Promise

How to determine object is a promise?

Check whether it has a `then()` method

### Arrow function

##### Without naming the function

One argument function

```javascript
// initial
function (a) {
  return a + 100;
}

// Arrow Function break down
// 1: remove "function" and place "=>"
(a) => {
  return a + 100;
}

// 2: remove "return" and curly braces
(a) => a + 100;

// 3: remove argument parentheses
a => a + 100;
```

No argument or multiple arguments function, need to re-introduce **(parenthese)**

```javascript
// Traditional: > 1 arguments
function (a,b) {
  return a + b + 100;
}
// Arrow
(a,b) => a + b + 100

// Traditional: 0 argument
let a = 10, b = 5;
function() {
  return a + b + 100;
}
// Arrow
() => a + b + 100;
```

**Additional Line** of processing, need to **PLUS the "return"** 

```javascript
// Traditional
function (a, b) {
  let chunk = 100;
  return a + b + chunk;
}
// Arrow
(a,b) => {
  let chunk = 100;
  return a + b + chunk;
}
```

##### Naming the function

```javascript
// Traditional
function foo(a) {
  return a + 100;
}
// Arrow
let foo = a => a + 100;
```

