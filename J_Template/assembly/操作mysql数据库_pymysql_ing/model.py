import pymysql


class MysqlDB(object):
    def __init__(self, host, user, password, db, port):
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        self.port = port
        self.connector = pymysql.connect(host=host, user=user, password=password, db=db, port=port)

    def fetch_data(self, sql):
        """获取数据"""
        cur = self.connector.cursor()
        cur.execute(sql)
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]
        cur.close()
        return data, columns

    def change_data(self, sql):
        """修改数据"""
        cur = self.connector.cursor()
        cur.execute(sql)
        self.connector.commit()
        cur.close()
