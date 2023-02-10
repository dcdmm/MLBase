```text
ctx._source(数据类型参考java.util.Map):
    Contains extracted JSON in a Map and List structure for the fields existing in a stored document.

doc:
    参考见:https://www.elastic.co/guide/en/elasticsearch/reference/7.11/modules-scripting-expression.html
    
    ===>获取text类型字段"content"的字符串文本:
    * doc['content'].value // 报错
    * 解决方法:
        1. 
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "fields": {
                        "raw": {
                            "type": "keyword"
                        }
                    }   
                }
            }
        2. doc['content.raw'].value // 此时正确
```

