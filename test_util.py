import re
def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    # 首先分割 英文 以及英文和标点
    pattern_char_1 = re.compile(r'([\W])')
    parts = pattern_char_1.split(sent)
    parts = [p for p in parts if len(p.strip())>0]
    # 分割中文
    result = []
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    for p in parts:
        chars = pattern.split(p)
        chars = [w for w in chars if len(w.strip())>0]
        result += chars
    return result


s = "啊今天是个good day啊!天气非常的nice，12sad 3123我想打110了"
print(seg_char(s))
