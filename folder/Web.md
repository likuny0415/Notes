

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

