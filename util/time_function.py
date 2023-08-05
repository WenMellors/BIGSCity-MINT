"""
temporal related function
"""
from datetime import datetime


def encode_timestamp(timestamp):
    """
    encode timestamp into discrete variables
    the time slot size is 5 minutes, different day is different
    :param timestamp: the timestamp, e.g. "2015-11-05T10:14:59Z"
    :return: time_code: code varying from 0 to 2015
    """
    # Convert provided string timestamp to datetime object
    time = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
    # Get weekday and minute from time
    weekday = time.weekday()
    minute = time.minute + time.hour * 60
    # Calculate number of time slots per day
    one_day_time_slot_nums = 24 * 60 // 5
    # Return calculated timecode
    return weekday * one_day_time_slot_nums + minute // 5


def encode_time(timestamp):
    """
    将字符串格式的时间戳编码 一分钟一个分桶
    """
    # 按一分钟编码，周末与工作日区分开来
    time = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
    if time.weekday() == 5 or time.weekday() == 6:
        return time.hour * 60 + time.minute + 1440
    else:
        return time.hour * 60 + time.minute


