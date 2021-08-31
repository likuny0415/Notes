# Media app

### GraphQL

- A Query language
- Support queries, mutations (subscriptions)
  - Queries help to fetch data
  - mutations help to update and create data
  - Subscriptions listen to actions
- Standardised specifiction
  - RESTFul API help to Post, Get resources



In GraphQL, every time we send a post, we will get something like this...

![image-20210821094208917](/Users/kuny/Library/Application Support/typora-user-images/image-20210821094208917.png)

But we want this ....

![image-20210821094336513](/Users/kuny/Library/Application Support/typora-user-images/image-20210821094336513.png)

Thus, GraphQL provides...

![image-20210821094443702](/Users/kuny/Library/Application Support/typora-user-images/image-20210821094443702.png)

Steps to Create code file

```terminal
$mkdir socialmeidaapp
$cd socialmediapp
$npm init
$touch index.js .gitignore
$git init
// add dependencies
$npm install apollo-server graphql mongoose
```

Things to be careful

- To highlight graphql download ```GraphQL for VSCode``` plugin
- Add token or important information into config.js
- When connect with MongoDB, need to change holder```<password>```&```databaseName``` in connection link 
-  ```./config``` Put ./ before config because it is a natural file, not a dependency

Create database

```javascript
// template with mongoose
const {model, Schema} = require('mongoose');

const nameSchema = new Schema({
  fieldName: type,
  fields..
  field: [
  	{
 		 	// field in field, e.g. comment -> which User, when
 		 	field: type,
 		 	fields...
		}
  ],
  // this allows to use Mongoose methods later
  user: {
    type: Schema.Types.ObjectId,
    refs: 'users'
  }
});

module.exports = model('modelName', nameSchema);
```









