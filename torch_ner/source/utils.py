import logging
import pickle
import json

class MyLogger(logging.Logger):

    def __init__(self, filename=None, stream=None):
        super(MyLogger, self).__init__("client_log")

        # Log等级总开关
        self.setLevel(logging.INFO)
        # 定义handler的输出格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

        if stream:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.addHandler(stream_handler)

        if filename:
            file_handler = logging.FileHandler(filename=filename, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)


def load_file(fp: str, sep: str = None, name_tuple=None):
    """
    读取文件；
    若sep为None，按行读取，返回文件内容列表，格式为:[xxx,xxx,xxx,...]
    若不为None，按行读取分隔，返回文件内容列表，格式为: [[xxx,xxx],[xxx,xxx],...]
    :param fp:
    :param sep:
    :param name_tuple:
    :return:
    """
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if sep:
            if name_tuple:
                return map(name_tuple._make, [line.strip().split(sep) for line in lines])
            else:
                return [line.strip().split(sep) for line in lines]
        else:
            return lines


def load_pkl(fp):
    """
    加载pkl文件
    :param fp:
    :return:
    """
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(data, fp):
    """
    保存pkl文件，数据序列化
    :param data:
    :param fp:
    :return:
    """
    with open(fp, 'wb') as f:
        pickle.dump(data, f)



def dump_json(obj, fp, encoding='utf-8', indent=4, ensure_ascii=False, json_lines=False):
    if not json_lines:
        with open(fp, 'w', encoding=encoding) as fout:
            json.dump(obj, fout, indent=indent, ensure_ascii=ensure_ascii)
    else:
        assert type(obj) == list
        with open(fp, "a+", encoding=encoding) as fout:
            for line in obj:
                fout.write(json.dumps(line)+"\n")


def load_json(fp, encoding='utf-8', json_lines=False):
    with open(fp, encoding=encoding) as fin:
        if not json_lines:
            return json.load(fin)
        else:
            ret = []
            for line in fin:
                ret.append(json.loads(line))
            return 
