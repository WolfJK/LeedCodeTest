# @Time   2020/8/15 15:15
# @Author Wolf
# @File   leetcode0815.py
import timeit
import numpy as np
def merge(array):
    '''
    [[1,3],[2,6],[8,10],[15,18]]
    :return: 合并区间  [[1,6],[8,10],[15,18]]
    '''
    resule = []
    if not array:
        return []
    array_sort = sorted(array, key=lambda x:x[0])
    resule.append(array_sort[0])
    for num in range(1, len(array_sort)):
        if array_sort[num][0] > resule[-1][1]:
            resule.append(array_sort[num])
        else:
            resule[-1][1] = max([resule[-1][1], array_sort[num][1]])

    print(resule)


def translate(array):
    '''转置二维数组'''
    '''
    transed= array[:]
    rows = len(array)
    cols = len(array[0])
    print('row-col', rows, cols)
    for i in range(cols):
        for j in range(rows):
            transed[i][j] = array[cols - j][i]
            print(array[cols - j][i])
    print(transed, id(array))
    '''

    array[:] = __import__("numpy").rot90(array, 3).tolist()
    print(array, )


def zero_all(array):
    '''将0所在的行与列均赋值为0'''
    df = np.array(array)
    print(df)
    rows = set(np.where(df==0)[0].tolist())
    cols = set(np.where(df==0)[1].tolist())
    print(rows, cols)
    for v in rows:
        print(v, 'v')
        df[v,:] = 0
    for v in cols:
        df[:,v] = 0
    print(df.tolist())


def longestCommonStr(common_str:list):
    '''
    编写一个函数来查找字符串数组中的最长公共前缀。
    如果不存在公共前缀，返回空字符串 ""。
    :param nums:
    :return: 不存在，返回 ""
    '''
    temp = ""
    if not common_str:
        return temp
    mins = common_str[0]
    l = len(common_str[0])
    # 查找最短的字符串
    for i in common_str:
        if len(i) < l:
            l = len(i)
            mins = i
    n = 0

    while n < len(mins):
        # print(compare)
        try:
            if not all(map(lambda x:x[n]==mins[n], common_str)):
                return temp
            temp += mins[n]
            n += 1
        except IndexError as e:
            print(n)
            return temp
    return temp


def reverseStr(strs):
    '''给定一个字符串，逐个翻转字符串中的每个单词'''
    if strs.isspace():
        return ""
    word_list = map(lambda x:"" if x == " " else x, reversed(strs.split()))
    print(list(reversed(strs.split())))

    return " ".join(word_list).strip()


def strStr(hay, need):
    """
    给定一个 haystack 字符串和一个 needle 字符串，
    在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。
    如果不存在，则返回  -1。
    """
    if not need:
        return 0
    data_index = {}
    for index, key in enumerate(hay):
        data_index.setdefault(key, []).append(index)
    lens = len(needle)
    first_str_index = data_index.get(need[0])
    if not first_str_index:
        return -1
    for index, str_base in enumerate(map(lambda x:hay[x:x+lens], first_str_index)):
        print(index, str_base)
        if str_base == need:
            return first_str_index[index]
    return -1

def strStrSecond(hay, need):
    if not need:
        return 0
    matched = 0
    n = 0
    start = 0
    first_index = 0
    for nn in __import__("re").finditer(need[0], hay):
        n = nn.span()[0]
        print(n, hay[n], "matched", matched, need[matched],"start", start)
        if hay[n] == need[matched]:
            if matched == 0 and first_index > 0:
                first_index = n
            matched += 1
            start = n
            if matched == len(need):
                return start - len(need) + 1
        else:
            matched = 0
            n, first_index = first_index, 0
        n += 1
    return -1


def strReversed(strs):
    """
    编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。
    不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
    你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符
    :param strs:
    :return:
    """
    left, right = 0, len(strs) - 1
    while left < right:
        strs[left], strs[right] = strs[right], strs[left]
        left += 1
        right -= 1
    print(strs)


def twoSum(numbers, target):
    """
    给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
    函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。
    说明:
        返回的下标值（index1 和 index2）不是从零开始的。
        你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。
    :param numbers:
    :param target:
    :return:
    """
    left = 0
    right = len(numbers)
    while True:
        if numbers[left] + numbers[right] > 0:
            pass


def generate(numRows):
    """
    给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
    :param numRows:  杨辉三角
    :return:
    """
    base = [[1]]
    if not numRows:
        return []
    while len(base) < numRows:
        rows_data = [i + j for i, j in zip([0] + base[-1], base[-1] + [0])]
        base.append(rows_data)
    return base

def generateSecond(numRows):
    """杨辉三角"""
    DATA = []
    if not numRows:
        return DATA

    def next_generate(num, datas):
        if num == 1:
            datas.append([1])
            return [1]
        d = []
        pre_nums = next_generate(num - 1, datas)
        for i in range(len(pre_nums) - 1):
            d.append(pre_nums[i] + pre_nums[i + 1])
        s = [1] + d + [1]
        datas.append(s)
        return s
    next_generate(numRows, DATA)
    return DATA


def TwoGenerate(rowIndex):
    """
    给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行。
    你可以优化你的算法到 O(k) 空间复杂度吗？
    :param rowIndex:
    :return:
    """
    base = [1]
    while len(base) < rowIndex + 1:
        nexta = [i + j for i, j in zip([0] + base, base + [0])]
        base = nexta
    return base


def reverseWords(s):
    '''给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序'''
    "Let's take LeetCode contest"
    temp = map(lambda x: "".join(reversed(x)), s.split())

    return " ".join(temp)


def findMin(nums):
    """
    假设按照升序排序的数组在预先未知的某个点上进行了旋转。
    ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
    请找出其中最小的元素。
    你可以假设数组中不存在重复元素。
    :param nums:
    :return:
    """
    if not nums:
        return 0
    base = float("-inf")
    for n in nums:
        if n >= base:
            base = n
        else:
            return n
    else:
        return nums[0]


def removeDuplicates(nums):
    '''
    给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
    不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成
    :param nums:
    :return:
    '''
    base = float("-inf")
    slow, fast = 0, 0
    while fast < len(nums):
        if float(nums[fast]) > base:
            base = float(nums[fast])
            nums[slow] = nums[fast]
            slow += 1
        fast += 1
    return len(nums[:slow])


def moveZeros(nums):
    '''
    给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序
    :param nums: array
    :return:
    '''
    fast = 0
    not_zero = 0
    while fast < len(nums):
        if nums[fast]:
            nums[not_zero], nums[fast] = nums[fast], nums[not_zero]
            not_zero += 1
        fast += 1


if __name__ == "__main__":
    array = [[1,1,1],[1,0,1],[1,1,1]]
    common_str = ["flower","flow","fight"]
    print("longestCommonStr", longestCommonStr(common_str),
          timeit.timeit("longestCommonStr(common_str)",
                        setup="from __main__ import longestCommonStr, common_str",
                        number=1))
    print(reverseStr("a good   example"))
    haystack = "mississippi"
    needle = "issip"
    print(strStr(haystack, needle))
    print("strReversed", strReversed(common_str))
    print(reverseWords("hello how are you?"))