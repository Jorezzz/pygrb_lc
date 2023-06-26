def find_closest_event(time,list_of_events):
    '''
    Return index of closest event to time by value
    Args:
        time ([float, datetime.datetime, datetime.date]): time to compare
        list_of_events (list): list of events to search in the same time format as 'time'
    '''
    res = min(list_of_events, key=lambda sub: abs(sub - time))
    index = list(list_of_events).index(res)
    return index

def is_intersected(left_1,right_1,left_2,right_2):
    '''
    Return True if intervals [left_1, right_1] and [left_2, right_2] are intersected
    Args:
        left_1 (float): left border of first interval
        right_1 (float): right border of first interval
        left_2 (float): left border of second interval
        right_2 (float): right border of second interval
    '''
    return (left_1 <= left_2 <= right_1) or \
            (left_1 <= right_2 <= right_1) or \
                (left_2 <= left_1 <= right_2) or \
                (left_2 <= right_1 <= right_2)