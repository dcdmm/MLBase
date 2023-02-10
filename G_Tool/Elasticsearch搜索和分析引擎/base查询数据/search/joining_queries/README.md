## Joining queries

Performing full SQL-style joins in a distributed system like Elasticsearch is prohibitively expensive. Instead,
Elasticsearch offers two forms of join which are designed to scale horizontally.

### nested query

Documents may contain fields of type nested. These fields are used to index arrays of objects, where each object can be
queried (with the nested query) as an independent document.

### has_child and has_parent queries

A join field relationship can exist between documents within a single index. The has_child query returns parent
documents whose child documents match the specified query, while the has_parent query returns child documents whose
parent document matches the specified query.