# -*- coding:utf-8 -*-
import pandas
import random
'''
题目1: 按照下面的格式打印1~10的数字以及每个数的平方、几何级数和阶乘
'''
def jiecheng(n):
	if n == 1:
		return 1
	else:
		return n * jiecheng(n - 1)
pandas.DataFrame({'数字':[_ for _ in range(1, 11)], '平方':[_ * _ for _ in range(1, 11)], '几何级数':[2 ** _ for _ in range(1, 11)], '阶乘':[jiecheng(_) for _ in range(1, 11)]})

'''
题目2: 设计一个函数，生成指定长度的验证码（由数字和大小写英文字母构成的随机字符串）
'''
def random_code(m):
    sd = [_ for _ in range(0, 10)]
    sd_str = [_ for _ in range(ord('A'), ord('Z'))] + [_ for _ in range(ord('a'), ord('z'))]
    new_str = ''
    for i in range(0, m):
        num_str = random.randint(0, 1)
        if num_str:
            str_index = chr(sd_str[random.randint(0, len(sd_str) - 1)])
        else:
            str_index = sd[random.randint(0, len(sd) - 1)]
        new_str += str(str_index)
    return new_str
# print(random_code(6))
'''
题目3: 设计一个函数，统计字符串中 英文字母和数字各自出现的次数 # Counter
'''

def str_num(abc):
    result = {}
    for i in abc:
        if (ord('0') <= ord(i.upper()) <= ord('9')):
            if i not in result.keys():
                result[i] = 1
            else:
                result[i] += 1
        elif (ord('A') < ord(i.upper()) < ord('Z')):
            if i.upper() not in result.keys():
                result[i] = 1
            else:
                result[i] += 1
    return result

# print(str_num('jhs98900$%^&*'))
'''
题目4: 设计一个函数，判断传入的整数列表（要求元素个数大于2个）中的元素能否构成等差数列
'''
def avg_num(num, agv):
    print(abs(num - agv))
def dc(num):

    cha = abs(num[0] - num[1])
    if len(num) > 2:
        not_ = 0
        s_num = sorted(num)
        for i in range(0, len(s_num) - 1):

            if abs(num[i] - num[i + 1]) != cha:
                not_ += 1
        print(num, '不是等差数列') if not_ else print(num, '是等差数列')
    else:
        print(num, '列表数据过少,不是等差数列')
num = [1, 4]

'''
题目5: 设计一个函数，计算字符串中所有数字序列的和
'''
def str_sum(abc):
    s = sum([int(_) for _ in abc if ord('0') <= ord(_) <= ord('9')])
    print(s)

# str_sum('21frs$2')
'''
题目6: 设计一个函数，对传入的字符串（假设字符串中只包含小写字母和空格）进行加密操作，
加密的规则是a变d，b变e，c变f，……，x变a，y变b，z变c，空格不变，返回加密后的字符串
'''
def md_str(abc):
    abc = abc.lower()
    new_ = ''
    for a in abc:
        if a == ' ':
            b = ' '
        else:
            b = chr(ord(a) + 3 if ord(a) + 3 <= ord('z') else ord('a') + ord(a) + 3 - ord('z') - 1)
        new_ += b
    print(new_)
# md_str('kkjaN lxyz')
'''
题目7: 设计“跳一跳”游戏的计分函数，“跳一跳”游戏中黑色小人从一个方块跳到另一个方块上会获得1分，
如果跳到方块的中心点上会获得2分，连续跳到中心点会依次获得2分、4分、6分、……。该函数传入一个列表，
列表中用布尔值True或False表示是否跳到方块的中心点，函数返回最后获得的分数
'''
locate = [True, False, False, True, True, False, True, False]
kl = {'False': 1, 'True': 2}

def grade_jump(locate):
    grade = 0
    for i in locate:
        if not i:
            grade1 = kl['False']
            kl['True'] = 2
        else:
            grade1 = kl['True']
            kl['True'] += 2
        grade += grade1
    print(grade)
# grade_jump(locate)
'''
题目8: 设计一个函数，统计一个字符串中出现频率最高的字符及其出现次数
'''
from collections import Counter
s = Counter('ksjasdnnwqwdfe')
# print(s, max(s.values()), s.most_common())
# print([_ for _ in s.most_common() if _[1] == s.most_common()[0][1]])


'''
题目9: 设计一个函数，传入两个代表日期的字符串，如“2018-2-26”、“2017-12-12”，计算两个日期相差多少天
'''
import time
import datetime
def max_diff(start_date, end_date):
    d1 = time.strptime(start_date, '%Y-%m-%d')
    d2 = time.strptime(end_date, '%Y-%m-%d')
    s1 = datetime.date(d1.tm_year, d1.tm_mon, d1.tm_mday)
    s2 = datetime.date(d2.tm_year, d2.tm_mon, d2.tm_mday)
    print(abs((s1 - s2).days))

max_diff('2018-2-26', '2017-12-12')
'''
题目10: “四川麻将”共有108张牌，分为“筒”、“条”、“万”三种类型，每种类型的牌有1~9的点数，
每个点数有4张牌；游戏通常有4个玩家，游戏开始的时候，有一个玩家会拿到14张牌（首家），
其他三个玩家每人13张牌。要求用面向对象的方式创建麻将牌和玩家并实现洗牌、摸牌以及玩家按照
类型和点数排列手上的牌的操作，最后显示出游戏开始的时候每个玩家手上的牌。
'''








