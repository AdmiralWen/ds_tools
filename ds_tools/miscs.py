"""
Miscellaneous Functions
Author: Brandon Wen

dfFromRSQuery: Creates a Pandas dataframe from a RedShift SQL query.
"""

import pandas as pd
import numpy as np
import boto3, json
import psycopg2, psycopg2.extras

def df_from_rs_query(env, rs_host, rs_port, rs_user, rs_password, query_string, timer = False):
    '''
    Creates a Pandas dataframe from a RedShift SQL query. The query_string argument should be your SQL string.

    Parameters:
    --------------------
    env: string
        The redshift database environment that you're querying from.
    rs_host: string
        The redshift host domain for your database.
    rs_port: string
        The port number for your query.
    rs_user: string
        Your database access username.
    rs_password: string
        Your database access password.
    query_string: string
        The SQL query you wish to execute.
    timer: boolean
        Specify whether you wish to print a runtime with your query.

    Returns:
    --------------------
    pandas.core.frame.DataFrame
    '''

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