"""
Miscellaneous Functions
Author: Brandon Wen

dfFromRSQuery: Creates a Pandas dataframe from a RedShift SQL query.
"""

import pandas as pd
import numpy as np

import boto3, json
import psycopg2, psycopg2.extras


def dfFromRSQuery(env, rs_host, rs_port, rs_user, rs_password, query_string, timer = False):
    ''' Creates a Pandas dataframe from a RedShift SQL query. The query_string argument should be your SQL string. '''
    t0 = time.time()
    con = psycopg2.connect(dbname=env, host=rs_host, port=rs_port, user=rs_user, password=rs_password)
    cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(query_string)
    response = cur.fetchall()
    dic = [dict(record) for record in response]
    cur.close()
    con.close()

    if timer:
        print("Query time:", time.time() - t0)
    return pd.DataFrame(dic)