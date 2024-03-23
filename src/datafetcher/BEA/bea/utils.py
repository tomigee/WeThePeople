def lowercase(data):
    if isinstance(data, dict):
        dic = {}
        for key in data:
            val = lowercase(data[key])
            dic[key.lower()] = val
        return dic
    elif isinstance(data, list):
        lst = []
        for item in data:
            val = lowercase(item)
            lst.append(val)
        return lst
    elif isinstance(data, str):
        return data.lower()
    else:
        return data
