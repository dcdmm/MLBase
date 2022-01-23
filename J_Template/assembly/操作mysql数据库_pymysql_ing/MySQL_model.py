import pymysql


class MysqlDB:
    def __init__(self, host, password, database, user='root', port=3306, charset='utf8'):
        # Host where the database server is located.
        self.host = host
        # Password to use.
        self.password = password
        # Database to use, None to not use a particular one.
        self.database = database
        # Username to log in as. (default: 'root')
        self.user = user
        #  MySQL port to use, default is usually OK. (default: 3306)
        self.port = port
        # Charset to use. (default: 'utf8')
        self.charset = charset
        # Establish a connection to the MySQL database
        self.connector = pymysql.connect(host=host, user=user, password=password, db=database, port=port,
                                         charset=charset)

    def fetch_data(self, sql):
        """
        获取数据---查询操作

        Parameters
        ---------
        sql : str
            MySQL语句

        Returns
        -------
        data : tuple
            查询结果
        columns: list
            查询结果的列名
        """
        cur = self.connector.cursor()
        # Execute a query.
        # Returns:  Number of affected rows.
        row_count = cur.execute(sql)
        print(row_count + " of affected rows!")
        # Fetch all the rows.
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]
        cur.close()
        return data, columns

    def change_data(self, sql, close=False):
        """
        修改数据---增、删、改

        Parameters
        ---------
        sql : str
            MySQL语句
        """
        cur = self.connector.cursor()
        row_count = cur.execute(sql)
        print(row_count + " of affected rows!")
        # Commit changes to stable storage
        self.connector.commit()
        cur.close()
        if close:
            self.connector.close()

    def close_con(self):
        """关闭数据连接"""
        self.connector.close()
